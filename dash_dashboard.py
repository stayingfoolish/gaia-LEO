# Dash Dashboard for Ground-Orbit Hybrid Infrastructure Scenario Planning
# =====================================================================
# Interactive web dashboard with visualizations and summary tables

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import pulp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our scenario planning model
from executive_summary_dashboard import ExecutiveSummaryDashboard

class DashScenarioDashboard:
    """
    Interactive Dash dashboard for ground-orbit hybrid infrastructure scenario planning.
    
    Features:
    - Interactive scenario selection
    - Real-time optimization results
    - Environmental impact visualizations
    - Cost structure analysis
    - Launch cadence impact analysis
    - Executive summary tables
    """
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.dashboard = ExecutiveSummaryDashboard()
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Set up the dashboard layout."""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 Ground-Orbit Hybrid Infrastructure Scenario Planning", 
                       style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': '30px'}),
                html.P("Interactive dashboard for analyzing orbital module feasibility, environmental impact, and cost structure",
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
                            {'label': 'Both Tight (CO₂ 10%, H₂O 60%)', 'value': 'both_tight'},
                            {'label': 'Both Soft (CO₂ 50%, H₂O 100%)', 'value': 'both_soft'}
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
                            {'label': 'No Launch Limits', 'value': None},
                            {'label': 'Weekly (52 modules/year)', 'value': 'weekly'},
                            {'label': 'Monthly (12 modules/year)', 'value': 'monthly'},
                            {'label': 'Quarterly (4 modules/year)', 'value': 'quarterly'}
                        ],
                        value=None,
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'backgroundColor': '#e9ecef', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '5px'}),
            
            # Summary Cards
            html.Div([
                html.Div([
                    html.H3("Total Demand", style={'color': '#2E86AB', 'margin': '0'}),
                    html.H2(id='total-demand', style={'color': '#2E86AB', 'margin': '0'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Orbit Share", style={'color': '#28a745', 'margin': '0'}),
                    html.H2(id='orbit-share', style={'color': '#28a745', 'margin': '0'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Orbital Modules", style={'color': '#ffc107', 'margin': '0'}),
                    html.H2(id='orbital-modules', style={'color': '#ffc107', 'margin': '0'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Total Cost", style={'color': '#dc3545', 'margin': '0'}),
                    html.H2(id='total-cost', style={'color': '#dc3545', 'margin': '0'})
                ], className='summary-card')
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
            
            # Main Content Tabs
            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='📊 Overview', value='overview'),
                dcc.Tab(label='🌍 Environmental Impact', value='environmental'),
                dcc.Tab(label='💰 Cost Analysis', value='cost'),
                dcc.Tab(label='🚀 Launch Impact', value='launch'),
                dcc.Tab(label='📋 Executive Summary', value='summary')
            ]),
            
            html.Div(id='tab-content', style={'marginTop': '20px'})
        ])
        
        # Add CSS styles
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Ground-Orbit Hybrid Infrastructure Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    .summary-card {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                        min-width: 200px;
                    }
                    .tab-content {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('total-demand', 'children'),
             Output('orbit-share', 'children'),
             Output('orbital-modules', 'children'),
             Output('total-cost', 'children')],
            [Input('demand-slider', 'value'),
             Input('resource-dropdown', 'value'),
             Input('launch-dropdown', 'value')]
        )
        def update_summary_cards(M, resource_scenario, launch_cadence):
            """Update summary cards based on scenario selection."""
            
            result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
            
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
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('demand-slider', 'value'),
             Input('resource-dropdown', 'value'),
             Input('launch-dropdown', 'value')]
        )
        def update_tab_content(active_tab, M, resource_scenario, launch_cadence):
            """Update tab content based on selection."""
            
            if active_tab == 'overview':
                return self.create_overview_tab(M, resource_scenario, launch_cadence)
            elif active_tab == 'environmental':
                return self.create_environmental_tab(M, resource_scenario, launch_cadence)
            elif active_tab == 'cost':
                return self.create_cost_tab(M, resource_scenario, launch_cadence)
            elif active_tab == 'launch':
                return self.create_launch_tab(M, resource_scenario, launch_cadence)
            elif active_tab == 'summary':
                return self.create_summary_tab(M, resource_scenario, launch_cadence)
        
    def create_overview_tab(self, M, resource_scenario, launch_cadence):
        """Create overview tab with key metrics and charts."""
        
        # Get scenario results
        result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
        
        if result['status'] != 'Optimal':
            return html.Div([
                html.H3("❌ Scenario Failed", style={'color': 'red'}),
                html.P("The selected scenario could not be solved. Please try different parameters.")
            ])
        
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
                    html.P(f"CO₂ Utilization: {result['CO2_utilization']:.1%}"),
                    html.P(f"Water Utilization: {result['H2O_utilization']:.1%}")
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H5("Cost Structure"),
                    html.P(f"Total Cost: ${result['total_cost']/1e9:.1f}B"),
                    html.P(f"CapEx Share: {result['capex_share']:.1%}"),
                    html.P(f"OpEx Share: {result['opex_share']:.1%}")
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ])
    
    def create_environmental_tab(self, M, resource_scenario, launch_cadence):
        """Create environmental impact tab."""
        
        # Get scenario results
        result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
        
        if result['status'] != 'Optimal':
            return html.Div([
                html.H3("❌ Scenario Failed", style={'color': 'red'}),
                html.P("The selected scenario could not be solved. Please try different parameters.")
            ])
        
        # Create environmental charts
        fig1 = go.Figure(data=[
            go.Bar(name='CO₂ Reduction', x=['Current Scenario'], y=[result['CO2_reduction']*100], 
                   marker_color='#dc3545'),
            go.Bar(name='Water Reduction', x=['Current Scenario'], y=[result['H2O_reduction']*100], 
                   marker_color='#17a2b8')
        ])
        fig1.update_layout(title='Environmental Impact Reduction (%)', height=400)
        
        fig2 = go.Figure(data=[
            go.Bar(name='CO₂ Utilization', x=['Current Scenario'], y=[result['CO2_utilization']*100], 
                   marker_color='#ffc107'),
            go.Bar(name='Water Utilization', x=['Current Scenario'], y=[result['H2O_utilization']*100], 
                   marker_color='#6f42c1')
        ])
        fig2.update_layout(title='Resource Utilization (%)', height=400)
        
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
                    html.P(f"CO₂ Baseline: {self.dashboard.CO2_base/1e6:.1f} Mt CO₂/year"),
                    html.P(f"CO₂ Current: {result['actual_CO2']/1e6:.1f} Mt CO₂/year"),
                    html.P(f"CO₂ Cap: {result['CO2_cap']/1e6:.1f} Mt CO₂/year" if result['CO2_cap'] else "CO₂ Cap: No limit")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H5("Water Impact"),
                    html.P(f"Water Baseline: {self.dashboard.H2O_base/1e9:.1f} billion L/year"),
                    html.P(f"Water Current: {result['actual_H2O']/1e9:.1f} billion L/year"),
                    html.P(f"Water Cap: {result['H2O_cap']/1e9:.1f} billion L/year" if result['H2O_cap'] else "Water Cap: No limit")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H5("Impact Summary"),
                    html.P(f"CO₂ Reduction: {result['CO2_reduction']:.1%}"),
                    html.P(f"Water Reduction: {result['H2O_reduction']:.1%}"),
                    html.P(f"Orbit Share: {result['orbit_share']:.1%}")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ])
    
    def create_cost_tab(self, M, resource_scenario, launch_cadence):
        """Create cost analysis tab."""
        
        # Get scenario results
        result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
        
        if result['status'] != 'Optimal':
            return html.Div([
                html.H3("❌ Scenario Failed", style={'color': 'red'}),
                html.P("The selected scenario could not be solved. Please try different parameters.")
            ])
        
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
        ])
    
    def create_launch_tab(self, M, resource_scenario, launch_cadence):
        """Create launch impact analysis tab."""
        
        # Get scenario results
        result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
        
        if result['status'] != 'Optimal':
            return html.Div([
                html.H3("❌ Scenario Failed", style={'color': 'red'}),
                html.P("The selected scenario could not be solved. Please try different parameters.")
            ])
        
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
                    html.P(f"Max Modules/Year: {self.dashboard.launch_cadences.get(launch_cadence, 'Unlimited') if launch_cadence else 'Unlimited'}"),
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
                    html.P(f"Launch Utilization: {result['total_modules']/self.dashboard.launch_cadences.get(launch_cadence, 1):.1%}" if launch_cadence else "Launch Utilization: N/A")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ])
    
    def create_summary_tab(self, M, resource_scenario, launch_cadence):
        """Create executive summary tab."""
        
        # Get scenario results
        result = self.dashboard.solve_scenario(M, resource_scenario, launch_cadence)
        
        if result['status'] != 'Optimal':
            return html.Div([
                html.H3("❌ Scenario Failed", style={'color': 'red'}),
                html.P("The selected scenario could not be solved. Please try different parameters.")
            ])
        
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
            {'Metric': 'CO₂ Utilization', 'Value': f"{result['CO2_utilization']:.1%}", 'Unit': '%'},
            {'Metric': 'Water Utilization', 'Value': f"{result['H2O_utilization']:.1%}", 'Unit': '%'}
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
                    html.P(f"Resource utilization: {result['CO2_utilization']:.1%} CO₂, {result['H2O_utilization']:.1%} water")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H5("💰 Cost Structure"),
                    html.P(f"Total cost: ${result['total_cost']/1e9:.1f}B"),
                    html.P(f"CapEx share: {result['capex_share']:.1%}"),
                    html.P(f"OpEx share: {result['opex_share']:.1%}")
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ])
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        print(f"🚀 Starting Ground-Orbit Hybrid Infrastructure Dashboard")
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print(f"🔧 Debug mode: {debug}")
        
        self.app.run(debug=debug, port=port)

# Create and run the dashboard
if __name__ == '__main__':
    dashboard = DashScenarioDashboard()
    dashboard.run(debug=True, port=8050)
