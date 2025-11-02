"""
Basic biochemical reaction models for testing and demonstration
"""

import numpy as np
from ..core.gillespie_gpu import GPUGillespieSimulator

class DimerizationModel:
    """
    Simple dimerization model: 2A ⇌ A₂
    """
    
    def __init__(self, 
                 k_dimerization: float = 0.1,
                 k_dissociation: float = 0.01,
                 initial_A: int = 1000,
                 initial_A2: int = 0):
        """
        Initialize dimerization model
        
        Parameters:
        -----------
        k_dimerization : float
            Dimerization rate constant
        k_dissociation : float
            Dissociation rate constant
        initial_A : int
            Initial count of monomer A
        initial_A2 : int
            Initial count of dimer A₂
        """
        self.k_dimerization = k_dimerization
        self.k_dissociation = k_dissociation
        
        # Define species and reactions
        species_names = ['A', 'A2']
        reaction_names = ['dimerization', 'dissociation']
        
        # Stoichiometry matrix
        # Rows: reactions, Columns: species
        # Negative values: reactants, Positive values: products
        stoichiometry = np.array([
            [-2,  1],  # 2A → A₂ (dimerization)
            [ 2, -1]   # A₂ → 2A (dissociation)
        ], dtype=np.int32)
        
        rate_constants = [k_dimerization, k_dissociation]
        
        initial_conditions = {
            'A': initial_A,
            'A2': initial_A2
        }
        
        # Create simulator
        self.simulator = GPUGillespieSimulator(
            species_names=species_names,
            reaction_names=reaction_names,
            stoichiometry=stoichiometry,
            rate_constants=rate_constants,
            initial_conditions=initial_conditions
        )
    
    def run_simulation(self, **kwargs):
        """Run simulation using the internal simulator"""
        return self.simulator.run_simulation(**kwargs)
    
    def get_equilibrium_constant(self) -> float:
        """Calculate theoretical equilibrium constant"""
        return self.k_dimerization / self.k_dissociation

class EnzymeKineticsModel:
    """
    Michaelis-Menten enzyme kinetics model
    E + S ⇌ ES → E + P
    """
    
    def __init__(self,
                 k_binding: float = 1.0,
                 k_dissociation: float = 0.1,
                 k_cat: float = 0.5,
                 initial_E: int = 100,
                 initial_S: int = 1000,
                 initial_ES: int = 0,
                 initial_P: int = 0):
        """
        Initialize enzyme kinetics model
        
        Parameters:
        -----------
        k_binding : float
            Enzyme-substrate binding rate
        k_dissociation : float
            ES complex dissociation rate
        k_cat : float
            Catalytic rate constant
        initial_E : int
            Initial enzyme count
        initial_S : int
            Initial substrate count
        initial_ES : int
            Initial ES complex count
        initial_P : int
            Initial product count
        """
        self.k_binding = k_binding
        self.k_dissociation = k_dissociation
        self.k_cat = k_cat
        
        # Species: E (enzyme), S (substrate), ES (complex), P (product)
        species_names = ['E', 'S', 'ES', 'P']
        reaction_names = ['binding', 'dissociation', 'catalysis']
        
        # Stoichiometry matrix
        stoichiometry = np.array([
            [-1, -1,  1,  0],  # E + S → ES
            [ 1,  1, -1,  0],  # ES → E + S
            [ 1,  0, -1,  1]   # ES → E + P
        ], dtype=np.int32)
        
        rate_constants = [k_binding, k_dissociation, k_cat]
        
        initial_conditions = {
            'E': initial_E,
            'S': initial_S,
            'ES': initial_ES,
            'P': initial_P
        }
        
        self.simulator = GPUGillespieSimulator(
            species_names=species_names,
            reaction_names=reaction_names,
            stoichiometry=stoichiometry,
            rate_constants=rate_constants,
            initial_conditions=initial_conditions
        )
    
    def run_simulation(self, **kwargs):
        """Run simulation using the internal simulator"""
        return self.simulator.run_simulation(**kwargs)
    
    def get_michaelis_constant(self) -> float:
        """Calculate Michaelis constant Km"""
        return (self.k_dissociation + self.k_cat) / self.k_binding
    
    def get_max_velocity(self, total_enzyme: int) -> float:
        """Calculate maximum velocity Vmax"""
        return self.k_cat * total_enzyme

class GeneExpressionModel:
    """
    Simple gene expression model with transcription and translation
    Gene → mRNA → Protein
    mRNA → ∅ (degradation)
    Protein → ∅ (degradation)
    """
    
    def __init__(self,
                 k_transcription: float = 0.1,
                 k_translation: float = 0.2,
                 k_mrna_degradation: float = 0.05,
                 k_protein_degradation: float = 0.01,
                 initial_gene: int = 1,
                 initial_mrna: int = 0,
                 initial_protein: int = 0):
        """
        Initialize gene expression model
        
        Parameters:
        -----------
        k_transcription : float
            Transcription rate
        k_translation : float
            Translation rate
        k_mrna_degradation : float
            mRNA degradation rate
        k_protein_degradation : float
            Protein degradation rate
        initial_gene : int
            Initial gene count (typically 1)
        initial_mrna : int
            Initial mRNA count
        initial_protein : int
            Initial protein count
        """
        self.k_transcription = k_transcription
        self.k_translation = k_translation
        self.k_mrna_degradation = k_mrna_degradation
        self.k_protein_degradation = k_protein_degradation
        
        # Species: Gene, mRNA, Protein
        species_names = ['Gene', 'mRNA', 'Protein']
        reaction_names = ['transcription', 'translation', 'mrna_degradation', 'protein_degradation']
        
        # Stoichiometry matrix
        stoichiometry = np.array([
            [ 0,  1,  0],  # Gene → Gene + mRNA (transcription)
            [ 0,  0,  1],  # mRNA → mRNA + Protein (translation)
            [ 0, -1,  0],  # mRNA → ∅ (degradation)
            [ 0,  0, -1]   # Protein → ∅ (degradation)
        ], dtype=np.int32)
        
        rate_constants = [
            k_transcription, 
            k_translation, 
            k_mrna_degradation, 
            k_protein_degradation
        ]
        
        initial_conditions = {
            'Gene': initial_gene,
            'mRNA': initial_mrna,
            'Protein': initial_protein
        }
        
        self.simulator = GPUGillespieSimulator(
            species_names=species_names,
            reaction_names=reaction_names,
            stoichiometry=stoichiometry,
            rate_constants=rate_constants,
            initial_conditions=initial_conditions
        )
    
    def run_simulation(self, **kwargs):
        """Run simulation using the internal simulator"""
        return self.simulator.run_simulation(**kwargs)
    
    def get_steady_state_mrna(self) -> float:
        """Calculate theoretical steady-state mRNA level"""
        return self.k_transcription / self.k_mrna_degradation
    
    def get_steady_state_protein(self) -> float:
        """Calculate theoretical steady-state protein level"""
        mrna_ss = self.get_steady_state_mrna()
        return self.k_translation * mrna_ss / self.k_protein_degradation

class ToggleSwitchModel:
    """
    Genetic toggle switch with mutual repression
    Promoter1 → Protein1 (repressed by Protein2)
    Promoter2 → Protein2 (repressed by Protein1)
    Protein1 → ∅ (degradation)
    Protein2 → ∅ (degradation)
    """
    
    def __init__(self,
                 k_baseline_1: float = 0.1,
                 k_baseline_2: float = 0.1,
                 k_repressed_1: float = 0.01,
                 k_repressed_2: float = 0.01,
                 k_cooperativity_1: float = 2.0,
                 k_cooperativity_2: float = 2.0,
                 k_hill_coeff_1: float = 2.0,
                 k_hill_coeff_2: float = 2.0,
                 k_degradation_1: float = 0.05,
                 k_degradation_2: float = 0.05,
                 initial_promoter1: int = 1,
                 initial_promoter2: int = 1,
                 initial_protein1: int = 10,
                 initial_protein2: int = 10):
        """
        Initialize toggle switch model
        
        Parameters describe Hill-type repression kinetics
        """
        self.k_baseline_1 = k_baseline_1
        self.k_baseline_2 = k_baseline_2
        self.k_repressed_1 = k_repressed_1
        self.k_repressed_2 = k_repressed_2
        self.k_cooperativity_1 = k_cooperativity_1
        self.k_cooperativity_2 = k_cooperativity_2
        self.k_hill_coeff_1 = k_hill_coeff_1
        self.k_hill_coeff_2 = k_hill_coeff_2
        self.k_degradation_1 = k_degradation_1
        self.k_degradation_2 = k_degradation_2
        
        # Simplified model with effective rates
        species_names = ['Promoter1', 'Promoter2', 'Protein1', 'Protein2']
        reaction_names = [
            'expression_1', 'expression_2',
            'degradation_1', 'degradation_2'
        ]
        
        # Simplified stoichiometry (effective model)
        stoichiometry = np.array([
            [ 0,  0,  1,  0],  # Promoter1 → Promoter1 + Protein1
            [ 0,  0,  0,  1],  # Promoter2 → Promoter2 + Protein2
            [ 0,  0, -1,  0],  # Protein1 → ∅
            [ 0,  0,  0, -1]   # Protein2 → ∅
        ], dtype=np.int32)
        
        # Effective rate constants (simplified for demonstration)
        rate_constants = [
            k_baseline_1,  # Effective expression rate 1
            k_baseline_2,  # Effective expression rate 2
            k_degradation_1,  # Degradation rate 1
            k_degradation_2   # Degradation rate 2
        ]
        
        initial_conditions = {
            'Promoter1': initial_promoter1,
            'Promoter2': initial_promoter2,
            'Protein1': initial_protein1,
            'Protein2': initial_protein2
        }
        
        self.simulator = GPUGillespieSimulator(
            species_names=species_names,
            reaction_names=reaction_names,
            stoichiometry=stoichiometry,
            rate_constants=rate_constants,
            initial_conditions=initial_conditions
        )
    
    def run_simulation(self, **kwargs):
        """Run simulation using the internal simulator"""
        return self.simulator.run_simulation(**kwargs)
    
    def get_bistability_condition(self) -> Dict:
        """
        Analyze bistability conditions
        Simplified analysis for the toggle switch
        """
        # This is a simplified analysis
        # Real bistability analysis would require more sophisticated methods
        ratio_1 = self.k_baseline_1 / self.k_degradation_1
        ratio_2 = self.k_baseline_2 / self.k_degradation_2
        
        return {
            'expression_degradation_ratio_1': ratio_1,
            'expression_degradation_ratio_2': ratio_2,
            'potential_bistable': ratio_1 > 1 and ratio_2 > 1
        }