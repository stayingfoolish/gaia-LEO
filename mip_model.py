#!/usr/bin/env python3
"""
Mixed-Integer Program (MIP) for Hybrid Space-Terrestrial Network Optimization

This module implements a comprehensive optimization model for deploying and operating
a hybrid satellite-terrestrial network to serve computational workloads.

Author: Generated for Gaia LEO Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import yaml
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NetworkOptimizer:
    """Main class for the hybrid space-terrestrial network optimization model."""
    
    def __init__(self, config_path: Optional[str] = None, data_dir: str = "./data", **cli_args):
        """Initialize the optimizer with configuration and CLI arguments."""
        self.config = self._load_config(config_path, cli_args)
        self.model = None
        self.data = {}
        self.solution = {}
        
        # Load data
        self.load_data(data_dir)
        
        # Create output directory
        Path("./outputs").mkdir(exist_ok=True)
        
    def _load_config(self, config_path: Optional[str], cli_args: Dict) -> Dict:
        """Load configuration from YAML file and override with CLI arguments."""
        default_config = {
            'time_limit_seconds': 600,
            'mip_gap': 0.02,
            'penalty_unserved': 1e9,
            'use_shuttles': False,  # Keep online-only
            'allow_ground_compute': True,
            'Lmax': 999,
            'solver': 'auto',
            # Starcloud paper cost parameters
            'p_space_energy': 0.002,  # $/kWh - orbital energy cost
            'p_ground_energy': 0.040,  # $/kWh - terrestrial energy cost
            'pue_orb': 1.20,  # Power Usage Effectiveness for orbital
            'pue_gs': 1.20,  # Power Usage Effectiveness for ground
            'ground_chiller_uplift': 0.05,  # +5% energy for ground cooling
            'capex_launch_per_module': 5_000_000,  # $5M per 40MW module
            'capex_shield_per_mw': 30_000,  # $30k/MW radiation shielding
            'capex_backup_ground_per_site': 20_000_000,  # $20M backup power per ground site
            'hours_per_period': 8760,  # 1 year planning period
            'amortize_capex_years': 0  # 0 = no amortization, >0 = annualize over N years
        }
        
        # Load from YAML if exists
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                default_config.update(yaml_config)
        
        # CLI arguments override config
        for key, value in cli_args.items():
            if value is not None:
                default_config[key] = value
                
        return default_config
    
    def load_data(self, inputs_dir: str = "./data") -> None:
        """Load all input data from CSV files."""
        inputs_path = Path(inputs_dir)
        
        # Load site data
        self.data['sites_ground'] = pd.read_csv(inputs_path / "sites_ground.csv")
        self.data['sites_orbit'] = pd.read_csv(inputs_path / "sites_orbit.csv")
        self.data['orbit_modules'] = pd.read_csv(inputs_path / "orbit_modules.csv")
        self.data['ground_costs'] = pd.read_csv(inputs_path / "ground_sites_costs.csv")
        self.data['links'] = pd.read_csv(inputs_path / "links.csv")
        self.data['jobs'] = pd.read_csv(inputs_path / "jobs.csv")
        
        # Optional shuttle data
        shuttle_path = inputs_path / "shuttles.csv"
        if shuttle_path.exists() and self.config['use_shuttles']:
            self.data['shuttles'] = pd.read_csv(shuttle_path)
        else:
            self.data['shuttles'] = pd.DataFrame()
            
        # Validate data
        self._validate_data()
        
        logger.info(f"Loaded data: {len(self.data['sites_ground'])} ground sites, "
                   f"{len(self.data['sites_orbit'])} orbit sites, "
                   f"{len(self.data['links'])} links, {len(self.data['jobs'])} jobs")
    
    def _validate_data(self) -> None:
        """Validate loaded data for consistency."""
        # Check that all links reference existing sites
        all_sites = set(self.data['sites_ground']['site_id']) | set(self.data['sites_orbit']['site_id'])
        link_sites = set(self.data['links']['src_id']) | set(self.data['links']['dst_id'])
        
        if not link_sites.issubset(all_sites):
            missing = link_sites - all_sites
            raise ValueError(f"Links reference unknown sites: {missing}")
        
        # Check shuttle data if present
        if not self.data['shuttles'].empty:
            shuttle_sites = set(self.data['shuttles']['src_id']) | set(self.data['shuttles']['dst_id'])
            if not shuttle_sites.issubset(all_sites):
                missing = shuttle_sites - all_sites
                raise ValueError(f"Shuttles reference unknown sites: {missing}")
    
    def build_model(self) -> None:
        """Build the Pyomo optimization model."""
        logger.info("Building optimization model...")
        
        self.model = pyo.ConcreteModel()
        
        # Create sets
        self._create_sets()
        
        # Create parameters
        self._create_parameters()
        
        # Create variables
        self._create_variables()
        
        # Create objective
        self._create_objective()
        
        # Create constraints
        self._create_constraints()
        
        logger.info("Model built successfully")
    
    def _create_sets(self) -> None:
        """Create model sets."""
        # Site sets
        self.model.I_orbit = pyo.Set(initialize=self.data['sites_orbit']['site_id'].tolist())
        self.model.J_ground = pyo.Set(initialize=self.data['sites_ground']['site_id'].tolist())
        self.model.N = pyo.Set(initialize=list(self.model.I_orbit) + list(self.model.J_ground))
        
        # Link sets
        links_df = self.data['links']
        self.model.A = pyo.Set(initialize=[(row['src_id'], row['dst_id']) for _, row in links_df.iterrows()])
        
        # Link type subsets
        self.model.A_SG = pyo.Set(initialize=[
            (row['src_id'], row['dst_id']) for _, row in links_df.iterrows() 
            if row['type'] == 'space_ground'
        ])
        self.model.A_SS = pyo.Set(initialize=[
            (row['src_id'], row['dst_id']) for _, row in links_df.iterrows() 
            if row['type'] == 'space_space'
        ])
        self.model.A_GG = pyo.Set(initialize=[
            (row['src_id'], row['dst_id']) for _, row in links_df.iterrows() 
            if row['type'] == 'ground_ground'
        ])
        
        # Job set
        self.model.K_jobs = pyo.Set(initialize=self.data['jobs']['job_id'].tolist())
        
        # Shuttle set (if applicable)
        if not self.data['shuttles'].empty:
            self.model.S_shuttles = pyo.Set(initialize=[
                (row['src_id'], row['dst_id']) for _, row in self.data['shuttles'].iterrows()
            ])
        else:
            self.model.S_shuttles = pyo.Set()
    
    def _create_parameters(self) -> None:
        """Create model parameters from loaded data."""
        # Link parameters
        links_df = self.data['links']
        self.model.cap_link = pyo.Param(
            self.model.A,
            within=pyo.NonNegativeReals,
            initialize={(row['src_id'], row['dst_id']): row['capacity_tbps'] 
                       for _, row in links_df.iterrows()}
        )
        self.model.lat_link = pyo.Param(
            self.model.A,
            within=pyo.NonNegativeReals,
            initialize={(row['src_id'], row['dst_id']): row['latency_ms'] 
                       for _, row in links_df.iterrows()}
        )
        self.model.cost_link = pyo.Param(
            self.model.A,
            within=pyo.NonNegativeReals,
            initialize={(row['src_id'], row['dst_id']): row['enable_cost_usd_per_period'] 
                       for _, row in links_df.iterrows()}
        )
        
        # Orbit module parameters
        modules_df = self.data['orbit_modules']
        self.model.module_mw = pyo.Param(
            self.model.I_orbit,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['module_mw'] for _, row in modules_df.iterrows()}
        )
        self.model.module_cost = pyo.Param(
            self.model.I_orbit,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['module_fixed_cost_usd'] for _, row in modules_df.iterrows()}
        )
        self.model.max_modules = pyo.Param(
            self.model.I_orbit,
            within=pyo.NonNegativeIntegers,
            initialize={row['site_id']: row['max_modules'] for _, row in modules_df.iterrows()}
        )
        self.model.cap_pwr = pyo.Param(
            self.model.I_orbit,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['max_compute_mw_power'] for _, row in modules_df.iterrows()}
        )
        self.model.cap_th = pyo.Param(
            self.model.I_orbit,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['max_compute_mw_thermal'] for _, row in modules_df.iterrows()}
        )
        
        # Orbital OpEx parameters from Starcloud paper (read from CSV)
        if 'p_space_energy' in modules_df.columns:
            self.model.space_energy_cost = pyo.Param(
                self.model.I_orbit,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['p_space_energy'] for _, row in modules_df.iterrows()}
            )
        else:
            self.model.space_energy_cost = pyo.Param(
                self.model.I_orbit,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: self.config['p_space_energy'] for _, row in modules_df.iterrows()}
            )
        
        if 'pue_orb' in modules_df.columns:
            self.model.space_pue = pyo.Param(
                self.model.I_orbit,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['pue_orb'] for _, row in modules_df.iterrows()}
            )
        else:
            self.model.space_pue = pyo.Param(
                self.model.I_orbit,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: self.config['pue_orb'] for _, row in modules_df.iterrows()}
            )
        
        # Ground site parameters
        ground_df = self.data['ground_costs']
        self.model.ground_fixed = pyo.Param(
            self.model.J_ground,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['fixed_cost_usd_per_period'] for _, row in ground_df.iterrows()}
        )
        self.model.ground_mw = pyo.Param(
            self.model.J_ground,
            within=pyo.NonNegativeReals,
            initialize={row['site_id']: row['max_compute_mw'] for _, row in ground_df.iterrows()}
        )
        
        # Ground termination capacity (if available)
        if 'max_terminate_tbps' in ground_df.columns:
            self.model.ground_term_cap = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['max_terminate_tbps'] for _, row in ground_df.iterrows()}
            )
        else:
            # Default to high capacity if not specified
            self.model.ground_term_cap = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: 1000.0 for _, row in ground_df.iterrows()}
            )
        
        # Ground OpEx parameters from Starcloud paper (read from CSV)
        if 'p_ground_energy' in ground_df.columns:
            self.model.ground_energy_cost = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['p_ground_energy'] for _, row in ground_df.iterrows()}
            )
        else:
            # Default to config value
            self.model.ground_energy_cost = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: self.config['p_ground_energy'] for _, row in ground_df.iterrows()}
            )
        
        if 'pue_gs' in ground_df.columns:
            self.model.ground_pue = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['pue_gs'] for _, row in ground_df.iterrows()}
            )
        else:
            self.model.ground_pue = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: self.config['pue_gs'] for _, row in ground_df.iterrows()}
            )
        
        if 'ground_chiller_uplift' in ground_df.columns:
            self.model.ground_chiller_uplift = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: row['ground_chiller_uplift'] for _, row in ground_df.iterrows()}
            )
        else:
            self.model.ground_chiller_uplift = pyo.Param(
                self.model.J_ground,
                within=pyo.NonNegativeReals,
                initialize={row['site_id']: self.config['ground_chiller_uplift'] for _, row in ground_df.iterrows()}
            )
        
        # Job parameters
        jobs_df = self.data['jobs']
        
        # Helper function to handle NaN values
        def safe_float(value, default=0.0):
            if pd.isna(value) or value is None:
                return default
            return float(value)
        
        def safe_int(value, default=0):
            if pd.isna(value) or value is None:
                return default
            return int(value)
        
        self.model.comp_mw = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row['compute_mw']) for _, row in jobs_df.iterrows()}
        )
        self.model.B_k = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row.get('bisection_tbps', 0)) for _, row in jobs_df.iterrows()}
        )
        self.model.bw_online = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row['online_bw_tbps']) for _, row in jobs_df.iterrows()}
        )
        self.model.Dmax = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row['max_rtt_ms']) for _, row in jobs_df.iterrows()}
        )
        self.model.allow_off = pyo.Param(
            self.model.K_jobs,
            within=pyo.Binary,
            initialize={row['job_id']: safe_int(row.get('allow_offline_bulk', False)) for _, row in jobs_df.iterrows()}
        )
        self.model.Qoff = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row.get('offline_pb_per_cycle', 0)) for _, row in jobs_df.iterrows()}
        )
        self.model.Doff = pyo.Param(
            self.model.K_jobs,
            within=pyo.NonNegativeReals,
            initialize={row['job_id']: safe_float(row.get('max_offline_latency_hours', 999)) for _, row in jobs_df.iterrows()}
        )
        
        # Shuttle parameters (if applicable)
        if not self.data['shuttles'].empty:
            shuttle_df = self.data['shuttles']
            self.model.bulk_cap = pyo.Param(
                self.model.S_shuttles,
                within=pyo.NonNegativeReals,
                initialize={(row['src_id'], row['dst_id']): row['bulk_pb_per_cycle'] 
                           for _, row in shuttle_df.iterrows()}
            )
            self.model.bulk_lat = pyo.Param(
                self.model.S_shuttles,
                within=pyo.NonNegativeReals,
                initialize={(row['src_id'], row['dst_id']): row['latency_hours'] 
                           for _, row in shuttle_df.iterrows()}
            )
            self.model.bulk_cost = pyo.Param(
                self.model.S_shuttles,
                within=pyo.NonNegativeReals,
                initialize={(row['src_id'], row['dst_id']): row['cycle_cost_usd'] 
                           for _, row in shuttle_df.iterrows()}
            )
        
        # Big-M constants
        self.model.M_flow = max(self.model.cap_link.values()) * 2
        self.model.M_lat = 1e6
        self.model.M_off = 1e6
        
        # Starcloud paper cost parameters (now read from CSV files)
        # Keep only scalar parameters that are not site-specific
        self.model.hours_per_period = self.config['hours_per_period']
        self.model.amortize_capex_years = self.config['amortize_capex_years']
        
        # Configuration parameters
        self.model.penalty_unserved = self.config['penalty_unserved']
        self.model.Lmax = self.config['Lmax']
        self.model.allow_ground_compute = self.config['allow_ground_compute']
    
    def _create_variables(self) -> None:
        """Create model decision variables."""
        # Module deployment
        self.model.x = pyo.Var(self.model.I_orbit, domain=pyo.NonNegativeIntegers)
        
        # Ground site activation
        self.model.y_ground = pyo.Var(self.model.J_ground, domain=pyo.Binary)
        
        # Link activation
        self.model.z = pyo.Var(self.model.A, domain=pyo.Binary)
        
        # Shuttle activation (if applicable)
        if not self.data['shuttles'].empty:
            self.model.s = pyo.Var(self.model.S_shuttles, domain=pyo.Binary)
        
        # Job assignment
        self.model.assign = pyo.Var(self.model.I_orbit, self.model.K_jobs, domain=pyo.Binary)
        self.model.ingress = pyo.Var(self.model.J_ground, self.model.K_jobs, domain=pyo.Binary)
        
        # Ground job placement (new)
        self.model.ground_assign = pyo.Var(self.model.J_ground, self.model.K_jobs, domain=pyo.Binary)
        
        # Flow variables
        self.model.f = pyo.Var(self.model.A, self.model.K_jobs, domain=pyo.NonNegativeReals)
        self.model.yarc = pyo.Var(self.model.A, self.model.K_jobs, domain=pyo.Binary)
        
        # Shuttle flow (if applicable)
        if not self.data['shuttles'].empty:
            self.model.q = pyo.Var(self.model.S_shuttles, self.model.K_jobs, domain=pyo.NonNegativeReals)
        
        # Unserved jobs
        self.model.unserved = pyo.Var(self.model.K_jobs, domain=pyo.Binary)
    
    def _create_objective(self) -> None:
        """Create the objective function using Starcloud paper cost model."""
        def objective_rule(model):
            # === CapEx (Capital Expenditure) ===
            
            # Orbital CapEx: Use module cost from CSV (already includes launch + shielding)
            orbital_capex = sum(
                model.module_cost[i] * model.x[i]
                for i in model.I_orbit
            )
            
            # Ground CapEx: Use fixed cost from CSV (already includes backup power)
            ground_capex = sum(
                model.ground_fixed[j] * model.y_ground[j]
                for j in model.J_ground
            )
            
            # Link CapEx
            link_capex = sum(
                model.cost_link[a, b] * model.z[a, b] 
                for (a, b) in model.A
            )
            
            # Total CapEx (with optional amortization)
            if model.amortize_capex_years > 0:
                capex = (orbital_capex + ground_capex + link_capex) / model.amortize_capex_years
            else:
                capex = orbital_capex + ground_capex + link_capex
            
            # === OpEx (Operational Expenditure) - Energy Costs ===
            
            # Compute assigned MW at each site
            # MW_orb[i] = sum_k comp_mw[k] * assign[i,k]
            # MW_gs[j] = sum_k comp_mw[k] * ground_assign[j,k]
            
            # Orbital energy OpEx: kWh = MW * PUE * 1000 * hours (site-specific)
            opex_orb_energy = sum(
                model.comp_mw[k] * model.assign[i, k] * model.space_pue[i] * 1000 * model.hours_per_period * model.space_energy_cost[i]
                for i in model.I_orbit for k in model.K_jobs
            )
            
            # Ground energy OpEx: kWh = MW * PUE * 1000 * hours * (1 + chiller_uplift) (site-specific)
            opex_gs_energy = sum(
                model.comp_mw[k] * model.ground_assign[j, k] * model.ground_pue[j] * 1000 * model.hours_per_period * 
                model.ground_energy_cost[j] * (1 + model.ground_chiller_uplift[j])
                for j in model.J_ground for k in model.K_jobs
            )
            
            # === Total Cost ===
            obj = capex + opex_orb_energy + opex_gs_energy
            
            # Add shuttle costs if applicable (keeping for compatibility)
            if not self.data['shuttles'].empty:
                obj += sum(model.bulk_cost[a, b] * model.s[a, b] for (a, b) in model.S_shuttles)
            
            # Add unserved penalty
            obj += model.penalty_unserved * sum(model.unserved[k] for k in model.K_jobs)
            
            return obj
        
        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    def _create_constraints(self) -> None:
        """Create all model constraints."""
        self._create_assignment_constraints()
        self._create_capacity_constraints()
        self._create_flow_constraints()
        self._create_latency_constraints()
        self._create_bisection_constraints()
        self._create_shuttle_constraints()
        self._create_deployment_constraints()
    
    def _create_assignment_constraints(self) -> None:
        """Assignment completeness constraints."""
        def assignment_completeness_rule(model, k):
            if model.allow_ground_compute:
                return (sum(model.assign[i, k] for i in model.I_orbit) + 
                       sum(model.ground_assign[j, k] for j in model.J_ground) + 
                       model.unserved[k] == 1)
            else:
                # Ground compute disabled - only orbital placement allowed
                return sum(model.assign[i, k] for i in model.I_orbit) + model.unserved[k] == 1
        
        def ingress_completeness_rule(model, k):
            # Only orbital jobs need ingress points (ground jobs are served locally)
            return sum(model.ingress[j, k] for j in model.J_ground) == sum(model.assign[i, k] for i in model.I_orbit)
        
        def ground_assignment_tie_rule(model, j, k):
            # Ground assignment requires ground site activation
            return model.ground_assign[j, k] <= model.y_ground[j]
        
        def ground_compute_disable_rule(model, j, k):
            # If ground compute is disabled, no ground assignments allowed
            if not model.allow_ground_compute:
                return model.ground_assign[j, k] == 0
            else:
                return pyo.Constraint.Skip
        
        self.model.assignment_completeness = pyo.Constraint(
            self.model.K_jobs, rule=assignment_completeness_rule
        )
        self.model.ingress_completeness = pyo.Constraint(
            self.model.K_jobs, rule=ingress_completeness_rule
        )
        self.model.ground_assignment_tie = pyo.Constraint(
            self.model.J_ground, self.model.K_jobs, rule=ground_assignment_tie_rule
        )
        self.model.ground_compute_disable = pyo.Constraint(
            self.model.J_ground, self.model.K_jobs, rule=ground_compute_disable_rule
        )
    
    def _create_capacity_constraints(self) -> None:
        """Compute capacity constraints."""
        def orbit_module_capacity_rule(model, i):
            return sum(model.comp_mw[k] * model.assign[i, k] for k in model.K_jobs) <= model.module_mw[i] * model.x[i]
        
        def orbit_power_capacity_rule(model, i):
            return sum(model.comp_mw[k] * model.assign[i, k] for k in model.K_jobs) <= model.cap_pwr[i]
        
        def orbit_thermal_capacity_rule(model, i):
            return sum(model.comp_mw[k] * model.assign[i, k] for k in model.K_jobs) <= model.cap_th[i]
        
        def orbit_max_modules_rule(model, i):
            return model.x[i] <= model.max_modules[i]
        
        def ground_activation_rule(model, j, k):
            return model.y_ground[j] >= model.ingress[j, k]
        
        def ground_compute_capacity_rule(model, j):
            return sum(model.comp_mw[k] * model.ground_assign[j, k] for k in model.K_jobs) <= model.ground_mw[j] * model.y_ground[j]
        
        def ground_termination_capacity_rule(model, j):
            return sum(model.bw_online[k] * model.ground_assign[j, k] for k in model.K_jobs) <= model.ground_term_cap[j] * model.y_ground[j]
        
        self.model.orbit_module_capacity = pyo.Constraint(
            self.model.I_orbit, rule=orbit_module_capacity_rule
        )
        self.model.orbit_power_capacity = pyo.Constraint(
            self.model.I_orbit, rule=orbit_power_capacity_rule
        )
        self.model.orbit_thermal_capacity = pyo.Constraint(
            self.model.I_orbit, rule=orbit_thermal_capacity_rule
        )
        self.model.orbit_max_modules = pyo.Constraint(
            self.model.I_orbit, rule=orbit_max_modules_rule
        )
        self.model.ground_activation = pyo.Constraint(
            self.model.J_ground, self.model.K_jobs, rule=ground_activation_rule
        )
        self.model.ground_compute_capacity = pyo.Constraint(
            self.model.J_ground, rule=ground_compute_capacity_rule
        )
        self.model.ground_termination_capacity = pyo.Constraint(
            self.model.J_ground, rule=ground_termination_capacity_rule
        )
    
    def _create_flow_constraints(self) -> None:
        """Online flow conservation and capacity constraints."""
        def flow_conservation_rule(model, n, k):
            if n in model.J_ground:
                # Ground nodes: supply if ingress for orbital job, demand if assigned orbit
                supply = model.bw_online[k] * model.ingress[n, k]
                demand = 0  # Ground nodes don't demand flow
            else:
                # Orbit nodes: demand if assigned
                supply = 0
                demand = model.bw_online[k] * model.assign[n, k]
            
            inflow = sum(model.f[a, n, k] for (a, b) in model.A if b == n)
            outflow = sum(model.f[n, b, k] for (a, b) in model.A if a == n)
            
            return inflow - outflow == supply - demand
        
        def link_capacity_rule(model, a, b):
            return sum(model.f[a, b, k] for k in model.K_jobs) <= model.cap_link[a, b] * model.z[a, b]
        
        def flow_tie_rule(model, a, b, k):
            return model.f[a, b, k] <= model.M_flow * model.yarc[a, b, k]
        
        def path_tie_rule(model, a, b, k):
            return model.yarc[a, b, k] <= model.z[a, b]
        
        self.model.flow_conservation = pyo.Constraint(
            self.model.N, self.model.K_jobs, rule=flow_conservation_rule
        )
        self.model.link_capacity = pyo.Constraint(
            self.model.A, rule=link_capacity_rule
        )
        self.model.flow_tie = pyo.Constraint(
            self.model.A, self.model.K_jobs, rule=flow_tie_rule
        )
        self.model.path_tie = pyo.Constraint(
            self.model.A, self.model.K_jobs, rule=path_tie_rule
        )
    
    def _create_latency_constraints(self) -> None:
        """Primary path selection and latency constraints."""
        def path_conservation_rule(model, n, k):
            if n in model.J_ground:
                supply = model.ingress[n, k]
                demand = 0
            else:
                supply = 0
                demand = model.assign[n, k]
            
            inflow = sum(model.yarc[a, n, k] for (a, b) in model.A if b == n)
            outflow = sum(model.yarc[n, b, k] for (a, b) in model.A if a == n)
            
            return inflow - outflow == supply - demand
        
        def latency_sla_rule(model, k):
            # Apply latency constraints to all jobs (ground jobs will have 0 flow/paths)
            return (sum(model.lat_link[a, b] * model.yarc[a, b, k] for (a, b) in model.A) <= 
                   model.Dmax[k] + model.M_lat * model.unserved[k])
        
        self.model.path_conservation = pyo.Constraint(
            self.model.N, self.model.K_jobs, rule=path_conservation_rule
        )
        self.model.latency_sla = pyo.Constraint(
            self.model.K_jobs, rule=latency_sla_rule
        )
    
    def _create_bisection_constraints(self) -> None:
        """Bisection bandwidth constraints at orbital and ground sites."""
        def orbit_bisection_capacity_rule(model, i):
            # Simplified bisection constraint - use a fixed high capacity
            max_bisection = 1000.0  # Tbps - high capacity assumption
            return sum(model.B_k[k] * model.assign[i, k] for k in model.K_jobs) <= max_bisection
        
        def ground_bisection_capacity_rule(model, j):
            # Simplified ground bisection constraint - use a fixed high capacity
            max_ground_bisection = 500.0  # Tbps - high capacity assumption for ground
            return sum(model.B_k[k] * model.ground_assign[j, k] for k in model.K_jobs) <= max_ground_bisection
        
        self.model.orbit_bisection_capacity = pyo.Constraint(
            self.model.I_orbit, rule=orbit_bisection_capacity_rule
        )
        self.model.ground_bisection_capacity = pyo.Constraint(
            self.model.J_ground, rule=ground_bisection_capacity_rule
        )
    
    def _create_shuttle_constraints(self) -> None:
        """Shuttle-related constraints (if applicable)."""
        if self.data['shuttles'].empty:
            return
        
        def shuttle_requirement_rule(model, k):
            if model.allow_off[k] == 1:
                assigned_to_orbit = sum(model.assign[i, k] for i in model.I_orbit)
                return (sum(model.q[a, b, k] for (a, b) in model.S_shuttles) >= 
                       model.Qoff[k] * assigned_to_orbit - model.M_off * model.unserved[k])
            else:
                return sum(model.q[a, b, k] for (a, b) in model.S_shuttles) == 0
        
        def shuttle_capacity_rule(model, a, b):
            return sum(model.q[a, b, k] for k in model.K_jobs) <= model.bulk_cap[a, b] * model.s[a, b]
        
        def shuttle_latency_rule(model, a, b, k):
            # Only allow shuttle arcs that meet latency requirements
            if model.bulk_lat[a, b] <= model.Doff[k]:
                return model.q[a, b, k] <= model.M_flow * model.s[a, b]
            else:
                return model.q[a, b, k] == 0
        
        self.model.shuttle_requirement = pyo.Constraint(
            self.model.K_jobs, rule=shuttle_requirement_rule
        )
        self.model.shuttle_capacity = pyo.Constraint(
            self.model.S_shuttles, rule=shuttle_capacity_rule
        )
        self.model.shuttle_latency = pyo.Constraint(
            self.model.S_shuttles, self.model.K_jobs, rule=shuttle_latency_rule
        )
    
    def _create_deployment_constraints(self) -> None:
        """Launch/deployment cadence constraints."""
        def launch_limit_rule(model):
            return sum(model.x[i] for i in model.I_orbit) <= model.Lmax
        
        self.model.launch_limit = pyo.Constraint(rule=launch_limit_rule)
    
    def solve(self, solver: str = None, time_limit: int = None, mip_gap: float = None) -> bool:
        """Solve the optimization model."""
        logger.info("Starting optimization...")
        
        # Use provided parameters or fall back to config
        solver_name = solver or self.config['solver']
        time_limit = time_limit or self.config['time_limit_seconds']
        mip_gap = mip_gap or self.config['mip_gap']
        if solver_name == 'auto':
            # Try open-source solvers first
            for solver in ['cbc', 'glpk', 'highs']:
                if SolverFactory(solver).available():
                    solver_name = solver
                    logger.info(f"Using solver: {solver}")
                    break
            else:
                raise RuntimeError("No suitable open-source solver found. Please install CBC, GLPK, or HiGHS.")
        
        solver = SolverFactory(solver_name)
        
        # Set solver options
        if solver_name == 'gurobi':
            solver.options['TimeLimit'] = time_limit
            solver.options['MIPGap'] = mip_gap
        elif solver_name == 'cbc':
            solver.options['seconds'] = time_limit
            solver.options['ratio'] = 1 + mip_gap
        elif solver_name == 'glpk':
            solver.options['tmlim'] = time_limit
            solver.options['mipgap'] = mip_gap
        elif solver_name == 'highs':
            solver.options['time_limit'] = time_limit
            solver.options['mip_abs_gap'] = mip_gap * 1000  # HiGHS uses absolute gap
        
        # Solve
        try:
            results = solver.solve(self.model, tee=True)
            
            if (results.solver.termination_condition == TerminationCondition.optimal or
                results.solver.termination_condition == TerminationCondition.feasible):
                logger.info("Optimization completed successfully")
                self._extract_solution()
                return True
            else:
                logger.error(f"Optimization failed: {results.solver.termination_condition}")
                return False
                
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return False
    
    def _extract_solution(self) -> None:
        """Extract solution from the solved model."""
        self.solution = {
            'objective': pyo.value(self.model.objective),
            'modules': {i: pyo.value(self.model.x[i]) for i in self.model.I_orbit},
            'ground_sites': {j: pyo.value(self.model.y_ground[j]) for j in self.model.J_ground},
            'links': {(a, b): pyo.value(self.model.z[a, b]) for (a, b) in self.model.A},
            'assignments': {(i, k): pyo.value(self.model.assign[i, k]) for i in self.model.I_orbit for k in self.model.K_jobs},
            'ground_assignments': {(j, k): pyo.value(self.model.ground_assign[j, k]) for j in self.model.J_ground for k in self.model.K_jobs},
            'ingress': {(j, k): pyo.value(self.model.ingress[j, k]) for j in self.model.J_ground for k in self.model.K_jobs},
            'flows': {(a, b, k): pyo.value(self.model.f[a, b, k]) for (a, b) in self.model.A for k in self.model.K_jobs},
            'paths': {(a, b, k): pyo.value(self.model.yarc[a, b, k]) for (a, b) in self.model.A for k in self.model.K_jobs},
            'unserved': {k: pyo.value(self.model.unserved[k]) for k in self.model.K_jobs}
        }
        
        if not self.data['shuttles'].empty:
            self.solution['shuttles'] = {(a, b): pyo.value(self.model.s[a, b]) for (a, b) in self.model.S_shuttles}
            self.solution['shuttle_flows'] = {(a, b, k): pyo.value(self.model.q[a, b, k]) for (a, b) in self.model.S_shuttles for k in self.model.K_jobs}
    
    def write_outputs(self) -> None:
        """Write solution outputs to CSV files and JSON summary."""
        logger.info("Writing outputs...")
        
        # Selected modules
        modules_data = []
        for site_id, x_val in self.solution['modules'].items():
            if x_val > 0:
                module_info = self.data['orbit_modules'][self.data['orbit_modules']['site_id'] == site_id].iloc[0]
                comp_cap = module_info['module_mw'] * x_val
                cap_bound = min(module_info['max_compute_mw_power'], 
                              module_info['max_compute_mw_thermal'], 
                              comp_cap)
                modules_data.append({
                    'site_id': site_id,
                    'x_modules': int(x_val),
                    'comp_cap_mw': comp_cap,
                    'cap_bound_mw': cap_bound
                })
        
        pd.DataFrame(modules_data).to_csv('./outputs/selected_modules.csv', index=False)
        
        # Enabled links
        links_data = []
        for (src, dst), enabled in self.solution['links'].items():
            if enabled > 0.5:
                link_info = self.data['links'][
                    (self.data['links']['src_id'] == src) & 
                    (self.data['links']['dst_id'] == dst)
                ].iloc[0]
                links_data.append({
                    'src_id': src,
                    'dst_id': dst,
                    'type': link_info['type'],
                    'capacity_tbps': link_info['capacity_tbps'],
                    'latency_ms': link_info['latency_ms'],
                    'enabled': 1
                })
        
        pd.DataFrame(links_data).to_csv('./outputs/enabled_links.csv', index=False)
        
        # Shuttle plan (if applicable)
        if 'shuttles' in self.solution:
            shuttle_data = []
            for (src, dst), enabled in self.solution['shuttles'].items():
                if enabled > 0.5:
                    total_pb = sum(self.solution['shuttle_flows'].get((src, dst, k), 0) for k in self.model.K_jobs)
                    shuttle_data.append({
                        'src_id': src,
                        'dst_id': dst,
                        'total_pb_per_cycle': total_pb,
                        'enabled': 1
                    })
            
            pd.DataFrame(shuttle_data).to_csv('./outputs/shuttle_plan.csv', index=False)
        
        # Job assignments
        job_data = []
        for k in self.model.K_jobs:
            assigned_orbit = None
            assigned_ground = None
            ingress_ground = None
            served = 1 - self.solution['unserved'][k]
            placement_type = 'unserved'
            
            if served > 0.5:
                # Check orbital assignment
                for i in self.model.I_orbit:
                    if self.solution['assignments'].get((i, k), 0) > 0.5:
                        assigned_orbit = i
                        placement_type = 'orbital'
                        break
                
                # Check ground assignment
                if assigned_orbit is None:
                    for j in self.model.J_ground:
                        if self.solution['ground_assignments'].get((j, k), 0) > 0.5:
                            assigned_ground = j
                            placement_type = 'ground'
                            break
                
                # Find ingress ground site
                for j in self.model.J_ground:
                    if self.solution['ingress'].get((j, k), 0) > 0.5:
                        ingress_ground = j
                        break
            
            # Calculate primary path RTT (only for orbital jobs)
            rtt_ms = 0
            if served > 0.5 and placement_type == 'orbital':
                for (a, b) in self.model.A:
                    if self.solution['paths'].get((a, b, k), 0) > 0.5:
                        rtt_ms += self.data['links'][
                            (self.data['links']['src_id'] == a) & 
                            (self.data['links']['dst_id'] == b)
                        ]['latency_ms'].iloc[0]
            
            job_info = self.data['jobs'][self.data['jobs']['job_id'] == k].iloc[0]
            job_data.append({
                'job_id': k,
                'placement_type': placement_type,
                'assigned_orbit': assigned_orbit,
                'assigned_ground': assigned_ground,
                'ingress_ground': ingress_ground,
                'served': int(served),
                'online_bw_tbps': job_info['online_bw_tbps'],
                'rtt_ms_primary_path': rtt_ms
            })
        
        pd.DataFrame(job_data).to_csv('./outputs/job_assignment.csv', index=False)
        
        # Job primary paths
        path_data = []
        for k in self.model.K_jobs:
            if self.solution['unserved'][k] < 0.5:
                seq = 1
                for (a, b) in self.model.A:
                    if self.solution['paths'].get((a, b, k), 0) > 0.5:
                        link_info = self.data['links'][
                            (self.data['links']['src_id'] == a) & 
                            (self.data['links']['dst_id'] == b)
                        ].iloc[0]
                        path_data.append({
                            'job_id': k,
                            'seq': seq,
                            'src_id': a,
                            'dst_id': b,
                            'latency_ms': link_info['latency_ms']
                        })
                        seq += 1
        
        pd.DataFrame(path_data).to_csv('./outputs/job_primary_paths.csv', index=False)
        
        # Summary JSON
        summary = {
            'objective': self.solution['objective'],
            'total_cost_breakdown': {
                'module_costs': sum(self.solution['modules'][i] * self.data['orbit_modules'][
                    self.data['orbit_modules']['site_id'] == i]['module_fixed_cost_usd'].iloc[0] 
                    for i in self.model.I_orbit),
                'ground_costs': sum(self.solution['ground_sites'][j] * self.data['ground_costs'][
                    self.data['ground_costs']['site_id'] == j]['fixed_cost_usd_per_period'].iloc[0] 
                    for j in self.model.J_ground),
                'link_costs': sum(self.solution['links'][(a, b)] * self.data['links'][
                    (self.data['links']['src_id'] == a) & (self.data['links']['dst_id'] == b)
                ]['enable_cost_usd_per_period'].iloc[0] for (a, b) in self.model.A)
            },
            'feasibility_flags': {
                'all_jobs_served': all(self.solution['unserved'][k] < 0.5 for k in self.model.K_jobs),
                'latency_violations': 0  # Would need to check against Dmax
            },
            'counts': {
                'enabled_modules': sum(1 for x in self.solution['modules'].values() if x > 0),
                'enabled_ground_sites': sum(1 for y in self.solution['ground_sites'].values() if y > 0.5),
                'enabled_links': sum(1 for z in self.solution['links'].values() if z > 0.5),
                'served_jobs': sum(1 for u in self.solution['unserved'].values() if u < 0.5),
                'total_jobs': len(self.model.K_jobs)
            }
        }
        
        with open('./outputs/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Outputs written successfully")
    
    def print_summary(self) -> None:
        """Print a concise summary of the solution with Starcloud paper cost breakdown."""
        if not self.solution:
            logger.error("No solution available")
            return
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY (Starcloud Paper Cost Model)")
        print("="*60)
        
        # Calculate cost components
        orbital_modules = sum(x for x in self.solution['modules'].values() if x > 0)
        ground_sites = sum(1 for y in self.solution['ground_sites'].values() if y > 0.5)
        enabled_links = sum(1 for z in self.solution['links'].values() if z > 0.5)
        
        # Calculate MW deployed
        orbital_mw = sum(
            self.model.comp_mw[k] * self.solution['assignments'].get((i, k), 0)
            for i in self.model.I_orbit for k in self.model.K_jobs
        )
        ground_mw = sum(
            self.model.comp_mw[k] * self.solution['ground_assignments'].get((j, k), 0)
            for j in self.model.J_ground for k in self.model.K_jobs
        )
        
        # CapEx breakdown
        orbital_capex = orbital_modules * (self.config['capex_launch_per_module'] + self.config['capex_shield_per_mw'] * 40)
        ground_capex = ground_sites * self.config['capex_backup_ground_per_site']
        link_capex = sum(
            self.model.cost_link[a, b] * self.solution['links'].get((a, b), 0)
            for (a, b) in self.model.A
        )
        total_capex = orbital_capex + ground_capex + link_capex
        
        # OpEx breakdown (energy)
        kwh_orb = orbital_mw * self.config['pue_orb'] * 1000 * self.config['hours_per_period']
        kwh_gs = ground_mw * self.config['pue_gs'] * 1000 * self.config['hours_per_period']
        opex_orb = self.config['p_space_energy'] * kwh_orb
        opex_gs = self.config['p_ground_energy'] * (1 + self.config['ground_chiller_uplift']) * kwh_gs
        total_opex = opex_orb + opex_gs
        
        print(f"\n📊 TOTAL COST: ${self.solution['objective']:,.2f}")
        print(f"\n💰 CapEx Breakdown:")
        print(f"  Orbital (Launch + Shielding): ${orbital_capex:,.2f}")
        print(f"  Ground (Backup Power):         ${ground_capex:,.2f}")
        print(f"  Network Links:                 ${link_capex:,.2f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total CapEx:                   ${total_capex:,.2f}")
        
        print(f"\n⚡ OpEx Breakdown (Energy, {self.config['hours_per_period']} hours):")
        print(f"  Orbital ({orbital_mw:.1f} MW @ ${self.config['p_space_energy']:.3f}/kWh): ${opex_orb:,.2f}")
        print(f"  Ground ({ground_mw:.1f} MW @ ${self.config['p_ground_energy']:.3f}/kWh):  ${opex_gs:,.2f}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total OpEx:                    ${total_opex:,.2f}")
        
        print(f"\n📈 Infrastructure:")
        print(f"  Orbital Modules:    {int(orbital_modules)}")
        print(f"  Ground Sites:       {ground_sites}")
        print(f"  Network Links:      {enabled_links}")
        
        print(f"\n🎯 Job Placement:")
        orbital_jobs = sum(1 for (i, k) in self.solution['assignments'] if self.solution['assignments'][(i, k)] > 0.5)
        ground_jobs = sum(1 for (j, k) in self.solution['ground_assignments'] if self.solution['ground_assignments'][(j, k)] > 0.5)
        served_jobs = sum(1 for u in self.solution['unserved'].values() if u < 0.5)
        print(f"  Orbital Jobs:       {orbital_jobs}")
        print(f"  Ground Jobs:        {ground_jobs}")
        print(f"  Served Jobs:        {served_jobs}/{len(self.model.K_jobs)}")
        
        print(f"\n📊 Cost per MW:")
        if orbital_mw > 0:
            print(f"  Orbital: ${(orbital_capex + opex_orb) / orbital_mw:,.2f}/MW")
        if ground_mw > 0:
            print(f"  Ground:  ${(ground_capex + opex_gs) / ground_mw:,.2f}/MW")
        
        print("="*60)


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Hybrid Space-Terrestrial Network Optimization')
    parser.add_argument('--solver', choices=['auto', 'cbc', 'glpk', 'highs'], default='auto',
                       help='Solver to use (default: auto)')
    parser.add_argument('--time_limit', type=int, default=600,
                       help='Time limit in seconds (default: 600)')
    parser.add_argument('--gap', type=float, default=0.02,
                       help='MIP gap tolerance (default: 0.02)')
    parser.add_argument('--use_shuttles', type=bool, default=False,
                       help='Enable shuttle constraints (default: False)')
    parser.add_argument('--allow_ground_compute', type=bool, default=True,
                       help='Allow ground compute placement (default: True)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--inputs_dir', type=str, default='./inputs',
                       help='Input directory path (default: ./inputs)')
    
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = NetworkOptimizer(
            config_path=args.config,
            solver=args.solver,
            time_limit_seconds=args.time_limit,
            mip_gap=args.gap,
            use_shuttles=args.use_shuttles
        )
        
        # Load data
        optimizer.load_data(args.inputs_dir)
        
        # Build and solve model
        optimizer.build_model()
        
        if optimizer.solve():
            optimizer.write_outputs()
            optimizer.print_summary()
        else:
            logger.error("Optimization failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
