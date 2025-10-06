# Simple Dash Dashboard for Ground-Orbit Hybrid Infrastructure
# ===========================================================
# Simplified version for easier debugging and deployment

import dash
from dash import dcc, html, Input, Output
import dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import pulp
import warnings
warnings.filterwarnings('ignore')

# Simple scenario planning model
class SimpleScenarioModel:
    """Simplified scenario planning model for the dashboard."""
    
    def __init__(self):
        self.setup_data()
        self.setup_parameters()
        
    def setup_data(self):
        """Set up the regional dataset."""
        # Updated costs based on detailed analysis
        # Ground: $0.04/kWh energy, $9M/40MW CapEx, includes cooling, backup, maintenance
        # Orbital: $0.002/kWh energy, $9M/40MW CapEx (launch + radiation shielding)
        self.summary = pd.DataFrame([
            ["US-East", 450, 0.040, 0.35, 0.30, 9.0],  # $0.04/kWh, $9M/40MW
            ["US-West", 350, 0.040, 0.25, 0.20, 9.0],  # $0.04/kWh, $9M/40MW  
            ["EU",      300, 0.040, 0.30, 0.15, 9.0],  # $0.04/kWh, $9M/40MW
            ["APAC",    400, 0.040, 0.50, 0.40, 9.0],  # $0.04/kWh, $9M/40MW
            ["LATAM",   200, 0.040, 0.45, 0.35, 9.0],  # $0.04/kWh, $9M/40MW
        ], columns=["region","cap_G","cost_G","e_G","w_G","capex_G"])
        
        for col in ["cap_G","cost_G","e_G","w_G","capex_G"]:
            self.summary[col] = pd.to_numeric(self.summary[col])
        
    def setup_parameters(self):
        """Set up global parameters."""
        self.alpha = 8760  # hours/year
        self.cost_O = 0.002  # $/kWh orbit energy
        self.f_O = 1464000  # $/yr per orbital module (to achieve $9M total CapEx)
        self.CapMod_O = 40  # MW per orbital module
        self.r, self.T = 0.10, 10  # discount rate, lifetime
        self.CRF = (self.r * (1 + self.r)**self.T) / ((1 + self.r)**self.T - 1)
        
        self.summary["annualized_capex"] = self.summary.capex_G * 1e6 * self.CRF
        self.baseline_capacity = self.summary.cap_G.sum()
        
        # Launch cadence options
        self.launch_cadences = {
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        
        # Resource cap scenarios
        self.resource_caps = {
            'no_caps': {'CO2_cap': None, 'H2O_cap': None},
            'CO2_10pct': {'CO2_cap': 0.10, 'H2O_cap': None},
            'CO2_50pct': {'CO2_cap': 0.50, 'H2O_cap': None},
            'H2O_60pct': {'CO2_cap': None, 'H2O_cap': 0.60},
            'both_tight': {'CO2_cap': 0.10, 'H2O_cap': 0.60},
            'both_soft': {'CO2_cap': 0.50, 'H2O_cap': 1.00}
        }
        
        # Calculate baselines
        self.CO2_base = (self.summary["cap_G"] * 1e3 * self.alpha * self.summary["e_G"]).sum()
        self.H2O_base = (self.summary["cap_G"] * 1e3 * self.alpha * self.summary["w_G"]).sum()
    
    def solve_scenario(self, M, resource_scenario, launch_rate=None):
        """Solve a single scenario."""
        try:
            # Calculate target demand
            D = M * self.baseline_capacity
            
            # Get resource caps
            caps = self.resource_caps[resource_scenario]
            CO2_cap = caps['CO2_cap'] * self.CO2_base if caps['CO2_cap'] else None
            H2O_cap = caps['H2O_cap'] * self.H2O_base if caps['H2O_cap'] else None
            
            # Handle launch rate (convert to None if 0 or None)
            if launch_rate == 0 or launch_rate is None:
                launch_rate = None
            
            # Create model
            model = pulp.LpProblem("SimpleScenario", pulp.LpMinimize)
            
            # Decision variables
            regions = self.summary.region.tolist()
            x_G = pulp.LpVariable.dicts("x_G", regions, lowBound=0, cat='Continuous')
            x_O = pulp.LpVariable("x_O", lowBound=0, cat='Continuous')
            b_G = pulp.LpVariable.dicts("b_G", regions, lowBound=0, cat='Continuous')
            m = pulp.LpVariable("m", lowBound=0, cat='Integer')
            
            # Objective
            energy_cost = pulp.lpSum([
                self.alpha * self.summary.loc[self.summary.region == j, 'cost_G'].iloc[0] * x_G[j]
                for j in regions
            ]) + self.alpha * self.cost_O * x_O
            
            capex_cost = pulp.lpSum([
                self.summary.loc[self.summary.region == j, 'annualized_capex'].iloc[0] * b_G[j]
                for j in regions
            ]) + self.f_O * m
            
            model += energy_cost + capex_cost
            
            # Constraints
            model += pulp.lpSum([x_G[j] for j in regions]) + x_O == D
            
            for j in regions:
                cap_G_j = self.summary.loc[self.summary.region == j, 'cap_G'].iloc[0]
                model += x_G[j] <= cap_G_j + b_G[j]
            
            model += x_O <= m * self.CapMod_O
            
            if launch_rate is not None:
                model += m <= launch_rate
            
            # Ground data center constraint: at least 1 ground for every 3 orbital modules
            # This ensures balanced deployment between ground and orbital infrastructure
            ground_count = pulp.lpSum([b_G[j] for j in regions])
            model += ground_count >= m / 3, "Ground_Orbit_Ratio"
            
            if CO2_cap is not None:
                CO2_ops = pulp.lpSum([
                    1e3 * self.alpha * self.summary.loc[self.summary.region == j, 'e_G'].iloc[0] * x_G[j]
                    for j in regions
                ])
                model += CO2_ops <= CO2_cap
            
            if H2O_cap is not None:
                H2O_ops = pulp.lpSum([
                    1e3 * self.alpha * self.summary.loc[self.summary.region == j, 'w_G'].iloc[0] * x_G[j]
                    for j in regions
                ])
                model += H2O_ops <= H2O_cap
            
            # Solve
            solver = pulp.PULP_CBC_CMD(msg=0)
            model.solve()
            
            if model.status == pulp.LpStatusOptimal:
                total_ground_load = sum(x_G[j].varValue for j in regions)
                total_orbit_load = x_O.varValue
                total_new_ground = sum(b_G[j].varValue for j in regions)
                total_modules = int(m.varValue)
                
                actual_CO2 = sum(
                    1e3 * self.alpha * self.summary.loc[self.summary.region == j, 'e_G'].iloc[0] * x_G[j].varValue
                    for j in regions
                )
                actual_H2O = sum(
                    1e3 * self.alpha * self.summary.loc[self.summary.region == j, 'w_G'].iloc[0] * x_G[j].varValue
                    for j in regions
                )
                
                ground_energy_cost = sum(
                    self.alpha * self.summary.loc[self.summary.region == j, 'cost_G'].iloc[0] * x_G[j].varValue
                    for j in regions
                )
                orbit_energy_cost = self.alpha * self.cost_O * total_orbit_load
                ground_capex_cost = sum(
                    self.summary.loc[self.summary.region == j, 'annualized_capex'].iloc[0] * b_G[j].varValue
                    for j in regions
                )
                orbit_capex_cost = self.f_O * total_modules
                
                return {
                    'status': 'Optimal',
                    'D': D,
                    'total_ground_load': total_ground_load,
                    'total_orbit_load': total_orbit_load,
                    'total_new_ground': total_new_ground,
                    'total_modules': total_modules,
                    'actual_CO2': actual_CO2,
                    'actual_H2O': actual_H2O,
                    'CO2_cap': CO2_cap,
                    'H2O_cap': H2O_cap,
                    'CO2_utilization': actual_CO2 / CO2_cap if CO2_cap else 0,
                    'H2O_utilization': actual_H2O / H2O_cap if H2O_cap else 0,
                    'CO2_used': actual_CO2,
                    'H2O_used': actual_H2O,
                    'ground_share': total_ground_load / D if D > 0 else 0,
                    'orbit_share': total_orbit_load / D if D > 0 else 0,
                    'CO2_reduction': (self.CO2_base - actual_CO2) / self.CO2_base if self.CO2_base > 0 else 0,
                    'H2O_reduction': (self.H2O_base - actual_H2O) / self.H2O_base if self.H2O_base > 0 else 0,
                    'ground_energy_cost': ground_energy_cost,
                    'orbit_energy_cost': orbit_energy_cost,
                    'ground_capex_cost': ground_capex_cost,
                    'orbit_capex_cost': orbit_capex_cost,
                    'total_cost': pulp.value(model.objective),
                    'total_opex': ground_energy_cost + orbit_energy_cost,
                    'total_capex': ground_capex_cost + orbit_capex_cost,
                    'capex_share': (ground_capex_cost + orbit_capex_cost) / pulp.value(model.objective) if pulp.value(model.objective) > 0 else 0,
                    'opex_share': (ground_energy_cost + orbit_energy_cost) / pulp.value(model.objective) if pulp.value(model.objective) > 0 else 0
                }
            else:
                return {'status': 'Failed', 'error': 'Optimization failed'}
                
        except Exception as e:
            return {'status': 'Failed', 'error': str(e)}

# Initialize the model
model = SimpleScenarioModel()

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚀 Ground-Orbit Hybrid Infrastructure Dashboard", 
               style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': '20px'}),
        html.P("Interactive scenario planning for orbital module feasibility and environmental impact",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Demand Multiplier (M):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='demand-slider',
                min=1,
                max=100,
                step=1,
                value=50,
                marks={i: str(i) for i in [1, 10, 20, 50, 100]},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Resource Scenario:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='resource-dropdown',
                options=[
                    {'label': 'No Constraints', 'value': 'no_caps'},
                    {'label': 'CO₂ 10% Cap', 'value': 'CO2_10pct'},
                    {'label': 'CO₂ 50% Cap', 'value': 'CO2_50pct'},
                    {'label': 'H₂O 60% Cap', 'value': 'H2O_60pct'},
                    {'label': 'Both Tight', 'value': 'both_tight'},
                    {'label': 'Both Soft', 'value': 'both_soft'}
                ],
                value='no_caps',
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Launch Cadence:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='launch-dropdown',
                options=[
                    {'label': 'No Launch Limits', 'value': 'no_limit'},
                    {'label': 'Weekly (52 modules/year)', 'value': 'weekly'},
                    {'label': 'Monthly (12 modules/year)', 'value': 'monthly'},
                    {'label': 'Quarterly (4 modules/year)', 'value': 'quarterly'}
                ],
                value='no_limit',
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ], style={'backgroundColor': '#e9ecef', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    # Summary Cards
    html.Div([
        html.Div([
            html.H3("Total Demand", style={'color': '#2E86AB', 'margin': '0'}),
            html.H2(id='total-demand', style={'color': '#2E86AB', 'margin': '0'})
        ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'minWidth': '200px'}),
        
        html.Div([
            html.H3("Orbit Share", style={'color': '#28a745', 'margin': '0'}),
            html.H2(id='orbit-share', style={'color': '#28a745', 'margin': '0'})
        ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'minWidth': '200px'}),
        
        html.Div([
            html.H3("Orbital Modules", style={'color': '#ffc107', 'margin': '0'}),
            html.H2(id='orbital-modules', style={'color': '#ffc107', 'margin': '0'})
        ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'minWidth': '200px'}),
        
        html.Div([
            html.H3("Total Cost", style={'color': '#dc3545', 'margin': '0'}),
            html.H2(id='total-cost', style={'color': '#dc3545', 'margin': '0'})
        ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'minWidth': '200px'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
        # Main Content Tabs
        dcc.Tabs(id='main-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='🌍 Environmental Impact', value='environmental'),
            dcc.Tab(label='💰 Cost Analysis', value='cost'),
            dcc.Tab(label='🚀 Launch Impact', value='launch'),
            dcc.Tab(label='📋 Executive Summary', value='summary')
        ], style={'marginTop': '20px'}),
        
        html.Div(id='tab-content', style={'marginTop': '20px'})
])

# Callbacks
@app.callback(
    [Output('total-demand', 'children'),
     Output('orbit-share', 'children'),
     Output('orbital-modules', 'children'),
     Output('total-cost', 'children')],
    [Input('demand-slider', 'value'),
     Input('resource-dropdown', 'value'),
     Input('launch-dropdown', 'value')]
)
def update_summary_cards(M, resource_scenario, launch_cadence):
    """Update summary cards."""
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] == 'Optimal':
        total_demand = f"{result['D']:,.0f} MW"
        orbit_share = f"{result['orbit_share']:.1%}"
        orbital_modules = f"{result['total_modules']:,}"
        total_cost = f"${result['total_cost']/1e9:.1f}B"
    else:
        total_demand = "Failed"
        orbit_share = "Failed"
        orbital_modules = "Failed"
        total_cost = "Failed"
    
    return total_demand, orbit_share, orbital_modules, total_cost

# Individual chart callbacks removed - now handled by tab system

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('demand-slider', 'value'),
     Input('resource-dropdown', 'value'),
     Input('launch-dropdown', 'value')]
)
def update_tab_content(active_tab, M, resource_scenario, launch_cadence):
    """Update tab content based on selection."""
    
    if active_tab == 'overview':
        return create_overview_tab(M, resource_scenario, launch_cadence)
    elif active_tab == 'environmental':
        return create_environmental_tab(M, resource_scenario, launch_cadence)
    elif active_tab == 'cost':
        return create_cost_tab(M, resource_scenario, launch_cadence)
    elif active_tab == 'launch':
        return create_launch_tab(M, resource_scenario, launch_cadence)
    elif active_tab == 'summary':
        return create_summary_tab(M, resource_scenario, launch_cadence)

def create_overview_tab(M, resource_scenario, launch_cadence):
    """Create overview tab with key metrics and charts."""
    
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    
    # Get scenario results
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] != 'Optimal':
        return html.Div([
            html.H3("❌ Scenario Failed", style={'color': 'red'}),
            html.P("The selected scenario could not be solved. Please try different parameters.")
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    # Create overview charts
    fig1 = go.Figure(data=[
        go.Bar(name='Ground Load', x=['Current Scenario'], y=[result['total_ground_load']], 
               marker_color='#2E86AB'),
        go.Bar(name='Orbit Load', x=['Current Scenario'], y=[result['total_orbit_load']], 
               marker_color='#28a745')
    ])
    fig1.update_layout(title='Load Distribution', barmode='stack', height=400)
    
    fig2 = go.Figure(data=[
        go.Pie(labels=['Ground', 'Orbit'], 
               values=[result['total_ground_load'], result['total_orbit_load']],
               marker_colors=['#2E86AB', '#28a745'])
    ])
    fig2.update_layout(title='Load Share Distribution', height=400)
    
    return html.Div([
        html.H3("📊 Scenario Overview"),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig1)
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig2)
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.H4("Key Metrics"),
        html.Div([
            html.Div([
                html.H5("Demand & Capacity"),
                html.P(f"Total Demand: {result['D']:,.0f} MW"),
                html.P(f"Ground Load: {result['total_ground_load']:,.0f} MW"),
                html.P(f"Orbit Load: {result['total_orbit_load']:,.0f} MW"),
                html.P(f"New Ground Capacity: {result['total_new_ground']:,.0f} MW")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Orbital Deployment"),
                html.P(f"Orbital Modules: {result['total_modules']:,}"),
                html.P(f"Orbit Share: {result['orbit_share']:.1%}"),
                html.P(f"Ground Share: {result['ground_share']:.1%}")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Environmental Impact"),
                html.P(f"CO₂ Reduction: {result['CO2_reduction']:.1%}"),
                html.P(f"Water Reduction: {result['H2O_reduction']:.1%}"),
                html.P(f"CO₂ Used: {result['CO2_used']/1e6:.1f} Mt CO₂/year"),
                html.P(f"Water Used: {result['H2O_used']/1e9:.1f} billion L/year")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Cost Structure"),
                html.P(f"Total Cost: ${result['total_cost']/1e9:.1f}B"),
                html.P(f"CapEx Share: {result['capex_share']:.1%}"),
                html.P(f"OpEx Share: {result['opex_share']:.1%}")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

def create_environmental_tab(M, resource_scenario, launch_cadence):
    """Create environmental impact tab."""
    
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    
    # Get scenario results
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] != 'Optimal':
        return html.Div([
            html.H3("❌ Scenario Failed", style={'color': 'red'}),
            html.P("The selected scenario could not be solved. Please try different parameters.")
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    # Create environmental charts
    fig1 = go.Figure(data=[
        go.Bar(name='CO₂ Reduction', x=['Current Scenario'], y=[result['CO2_reduction']*100], 
               marker_color='#dc3545'),
        go.Bar(name='Water Reduction', x=['Current Scenario'], y=[result['H2O_reduction']*100], 
               marker_color='#17a2b8')
    ])
    fig1.update_layout(title='Environmental Impact Reduction (%)', height=400)
    
    fig2 = go.Figure(data=[
        go.Bar(name='CO₂ Used', x=['Current Scenario'], y=[result['CO2_used']/1e6], 
               marker_color='#ffc107'),
        go.Bar(name='Water Used', x=['Current Scenario'], y=[result['H2O_used']/1e9], 
               marker_color='#6f42c1')
    ])
    fig2.update_layout(title='Resource Usage (Mt CO₂/year, billion L/year)', height=400)
    
    return html.Div([
        html.H3("🌍 Environmental Impact Analysis"),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig1)
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig2)
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.H4("Environmental Metrics"),
        html.Div([
            html.Div([
                html.H5("Baseline vs Current"),
                html.P(f"CO₂ Baseline: {model.CO2_base/1e6:.1f} Mt CO₂/year"),
                html.P(f"CO₂ Current: {result['actual_CO2']/1e6:.1f} Mt CO₂/year"),
                html.P(f"CO₂ Cap: {result['CO2_cap']/1e6:.1f} Mt CO₂/year" if result['CO2_cap'] else "CO₂ Cap: No limit")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Water Impact"),
                html.P(f"Water Baseline: {model.H2O_base/1e9:.1f} billion L/year"),
                html.P(f"Water Current: {result['actual_H2O']/1e9:.1f} billion L/year"),
                html.P(f"Water Cap: {result['H2O_cap']/1e9:.1f} billion L/year" if result['H2O_cap'] else "Water Cap: No limit")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Impact Summary"),
                html.P(f"CO₂ Reduction: {result['CO2_reduction']:.1%}"),
                html.P(f"Water Reduction: {result['H2O_reduction']:.1%}"),
                html.P(f"CO₂ Used: {result['CO2_used']/1e6:.1f} Mt CO₂/year"),
                html.P(f"Water Used: {result['H2O_used']/1e9:.1f} billion L/year"),
                html.P(f"Orbit Share: {result['orbit_share']:.1%}")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

def create_cost_tab(M, resource_scenario, launch_cadence):
    """Create cost analysis tab."""
    
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    
    # Get scenario results
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] != 'Optimal':
        return html.Div([
            html.H3("❌ Scenario Failed", style={'color': 'red'}),
            html.P("The selected scenario could not be solved. Please try different parameters.")
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    # Create cost charts
    fig1 = go.Figure(data=[
        go.Bar(name='Ground Energy', x=['Current Scenario'], y=[result['ground_energy_cost']/1e9], 
               marker_color='#2E86AB'),
        go.Bar(name='Orbit Energy', x=['Current Scenario'], y=[result['orbit_energy_cost']/1e9], 
               marker_color='#28a745'),
        go.Bar(name='Ground CapEx', x=['Current Scenario'], y=[result['ground_capex_cost']/1e9], 
               marker_color='#ffc107'),
        go.Bar(name='Orbit CapEx', x=['Current Scenario'], y=[result['orbit_capex_cost']/1e9], 
               marker_color='#dc3545')
    ])
    fig1.update_layout(title='Cost Breakdown ($B)', barmode='stack', height=400)
    
    fig2 = go.Figure(data=[
        go.Pie(labels=['CapEx', 'OpEx'], 
               values=[result['total_capex'], result['total_opex']],
               marker_colors=['#ffc107', '#2E86AB'])
    ])
    fig2.update_layout(title='CapEx vs OpEx Share', height=400)
    
    return html.Div([
        html.H3("💰 Cost Structure Analysis"),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig1)
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig2)
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.H4("Cost Breakdown"),
        html.Div([
            html.Div([
                html.H5("Energy Costs"),
                html.P(f"Ground Energy: ${result['ground_energy_cost']/1e9:.1f}B"),
                html.P(f"Orbit Energy: ${result['orbit_energy_cost']/1e9:.1f}B"),
                html.P(f"Total OpEx: ${result['total_opex']/1e9:.1f}B")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Capital Costs"),
                html.P(f"Ground CapEx: ${result['ground_capex_cost']/1e9:.1f}B"),
                html.P(f"Orbit CapEx: ${result['orbit_capex_cost']/1e9:.1f}B"),
                html.P(f"Total CapEx: ${result['total_capex']/1e9:.1f}B")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Cost Structure"),
                html.P(f"Total Cost: ${result['total_cost']/1e9:.1f}B"),
                html.P(f"CapEx Share: {result['capex_share']:.1%}"),
                html.P(f"OpEx Share: {result['opex_share']:.1%}")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Cost Efficiency"),
                html.P(f"Cost per MW: ${result['total_cost']/result['D']/1e6:.1f}M/MW"),
                html.P(f"Orbit Cost per MW: ${result['orbit_capex_cost']/result['total_orbit_load']/1e6:.1f}M/MW" if result['total_orbit_load'] > 0 else "Orbit Cost per MW: N/A"),
                html.P(f"Ground Cost per MW: ${result['ground_capex_cost']/result['total_ground_load']/1e6:.1f}M/MW" if result['total_ground_load'] > 0 else "Ground Cost per MW: N/A")
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

def create_launch_tab(M, resource_scenario, launch_cadence):
    """Create launch impact analysis tab."""
    
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    
    # Get scenario results
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] != 'Optimal':
        return html.Div([
            html.H3("❌ Scenario Failed", style={'color': 'red'}),
            html.P("The selected scenario could not be solved. Please try different parameters.")
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    # Create launch impact charts
    fig1 = go.Figure(data=[
        go.Bar(name='Orbital Modules', x=['Current Scenario'], y=[result['total_modules']], 
               marker_color='#28a745')
    ])
    fig1.update_layout(title='Orbital Modules Deployed', height=400)
    
    fig2 = go.Figure(data=[
        go.Bar(name='Orbit Share', x=['Current Scenario'], y=[result['orbit_share']*100], 
               marker_color='#2E86AB')
    ])
    fig2.update_layout(title='Orbit Share (%)', height=400)
    
    return html.Div([
        html.H3("🚀 Launch Cadence Impact Analysis"),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig1)
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig2)
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.H4("Launch Impact Metrics"),
        html.Div([
            html.Div([
                html.H5("Launch Constraints"),
                html.P(f"Launch Cadence: {launch_cadence if launch_cadence else 'No limits'}"),
                html.P(f"Max Modules/Year: {model.launch_cadences.get(launch_cadence, 'Unlimited') if launch_cadence else 'Unlimited'}"),
                html.P(f"Modules Deployed: {result['total_modules']:,}")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Deployment Impact"),
                html.P(f"Orbit Share: {result['orbit_share']:.1%}"),
                html.P(f"Ground Share: {result['ground_share']:.1%}"),
                html.P(f"New Ground Capacity: {result['total_new_ground']:,.0f} MW")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Cost Impact"),
                html.P(f"Total Cost: ${result['total_cost']/1e9:.1f}B"),
                html.P(f"Cost per Module: ${result['orbit_capex_cost']/result['total_modules']/1e6:.1f}M" if result['total_modules'] > 0 else "Cost per Module: N/A"),
                html.P(f"Launch Utilization: {result['total_modules']/model.launch_cadences.get(launch_cadence, 1):.1%}" if launch_cadence else "Launch Utilization: N/A")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

def create_summary_tab(M, resource_scenario, launch_cadence):
    """Create executive summary tab."""
    
    # Convert launch cadence to launch rate
    if launch_cadence == 'no_limit':
        launch_rate = None
    else:
        launch_rate = model.launch_cadences.get(launch_cadence) if launch_cadence else None
    
    # Get scenario results
    result = model.solve_scenario(M, resource_scenario, launch_rate)
    
    if result['status'] != 'Optimal':
        return html.Div([
            html.H3("❌ Scenario Failed", style={'color': 'red'}),
            html.P("The selected scenario could not be solved. Please try different parameters.")
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    # Create summary table
    summary_data = [
        {'Metric': 'Total Demand', 'Value': f"{result['D']:,.0f} MW", 'Unit': 'MW'},
        {'Metric': 'Ground Load', 'Value': f"{result['total_ground_load']:,.0f} MW", 'Unit': 'MW'},
        {'Metric': 'Orbit Load', 'Value': f"{result['total_orbit_load']:,.0f} MW", 'Unit': 'MW'},
        {'Metric': 'Orbital Modules', 'Value': f"{result['total_modules']:,}", 'Unit': 'modules'},
        {'Metric': 'Orbit Share', 'Value': f"{result['orbit_share']:.1%}", 'Unit': '%'},
        {'Metric': 'Ground Share', 'Value': f"{result['ground_share']:.1%}", 'Unit': '%'},
        {'Metric': 'Total Cost', 'Value': f"${result['total_cost']/1e9:.1f}B", 'Unit': '$B'},
        {'Metric': 'CapEx Share', 'Value': f"{result['capex_share']:.1%}", 'Unit': '%'},
        {'Metric': 'OpEx Share', 'Value': f"{result['opex_share']:.1%}", 'Unit': '%'},
        {'Metric': 'CO₂ Reduction', 'Value': f"{result['CO2_reduction']:.1%}", 'Unit': '%'},
        {'Metric': 'Water Reduction', 'Value': f"{result['H2O_reduction']:.1%}", 'Unit': '%'},
        {'Metric': 'CO₂ Used', 'Value': f"{result['CO2_used']/1e6:.1f} Mt CO₂/year", 'Unit': 'Mt CO₂/year'},
        {'Metric': 'Water Used', 'Value': f"{result['H2O_used']/1e9:.1f} billion L/year", 'Unit': 'billion L/year'}
    ]
    
    summary_df = pd.DataFrame(summary_data)
    
    return html.Div([
        html.H3("📋 Executive Summary"),
        
        html.H4("Scenario Parameters"),
        html.Div([
            html.Div([
                html.P(f"Demand Multiplier: {M}x"),
                html.P(f"Resource Scenario: {resource_scenario}"),
                html.P(f"Launch Cadence: {launch_cadence if launch_cadence else 'No limits'}")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        html.H4("Key Metrics Summary"),
        dash_table.DataTable(
            data=summary_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#2E86AB', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ]
        ),
        
        html.H4("Strategic Insights"),
        html.Div([
            html.Div([
                html.H5("🚀 Orbital Feasibility"),
                html.P(f"Orbital modules become feasible at {M}x demand multiplier"),
                html.P(f"Current deployment: {result['total_modules']:,} modules"),
                html.P(f"Orbit share: {result['orbit_share']:.1%}")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("🌍 Environmental Impact"),
                html.P(f"CO₂ reduction: {result['CO2_reduction']:.1%}"),
                html.P(f"Water reduction: {result['H2O_reduction']:.1%}"),
                html.P(f"Resource usage: {result['CO2_used']/1e6:.1f} Mt CO₂/year, {result['H2O_used']/1e9:.1f} billion L/year")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("💰 Cost Structure"),
                html.P(f"Total cost: ${result['total_cost']/1e9:.1f}B"),
                html.P(f"CapEx share: {result['capex_share']:.1%}"),
                html.P(f"OpEx share: {result['opex_share']:.1%}")
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

# Launch cadence dropdown is now handled directly in the layout

if __name__ == '__main__':
    print("🚀 Starting Simple Ground-Orbit Hybrid Infrastructure Dashboard")
    print("📊 Dashboard will be available at: http://localhost:8050")
    app.run(debug=True, port=8050)
