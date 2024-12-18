from absl import flags
from alphapulldown.folding_backend.alphafold_backend import ModelsToRelax

def define_alphafold_flags():
    # AlphaFold flags
    flags.DEFINE_integer('num_cycle', 3,
                         'Number of recycles, defaults to 3.')
    flags.DEFINE_integer('num_predictions_per_model', 1,
                         'Number of predictions per model, defaults to 1.')
    flags.DEFINE_boolean('pair_msa', True,
                         'Whether to pair MSAs for multimer objects.')
    flags.DEFINE_boolean('skip_templates', False,
                         'Do not use template features when modelling.')
    flags.DEFINE_boolean('msa_depth_scan', False,
                         'Run predictions with logarithmically distributed MSA depth.')
    flags.DEFINE_boolean('multimeric_template', False,
                         'Whether to use multimeric templates.')
    flags.DEFINE_list('model_names', None,
                      'Names of models to use (default: all models).')
    flags.DEFINE_integer('msa_depth', None,
                         'Number of sequences to use from the MSA.')
    flags.DEFINE_string('description_file', None,
                        'Path to multimeric template instruction file.')
    flags.DEFINE_string('path_to_mmt', None,
                        'Path to directory with multimeric template mmCIF files.')
    flags.DEFINE_integer('desired_num_res', None,
                         'Number of residues to pad.')
    flags.DEFINE_integer('desired_num_msa', None,
                         'Number of MSAs to pad.')
    flags.DEFINE_enum_class(
        "models_to_relax",
        ModelsToRelax.NONE,
        ModelsToRelax,
        "Models to run final relaxation step on."
    )
    flags.DEFINE_enum('model_preset', 'monomer',
                      ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                      'Model configuration preset.')
    flags.DEFINE_boolean('benchmark', False,
                         'Benchmark mode.')
    flags.DEFINE_integer('random_seed', None,
                         'Random seed for data pipeline.')
    flags.DEFINE_boolean('convert_to_modelcif', True,
                         'Convert predicted pdb files to modelcif format.')
    flags.DEFINE_boolean('allow_resume', True,
                         'Allow resuming predictions.')
    flags.DEFINE_string('crosslinks', None,
                        'Path to crosslink information pickle (for AlphaLink/AlphaFold).')

def define_alphafold3_flags():
    # AlphaFold3-specific flags
    flags.DEFINE_string(
        'jax_compilation_cache_dir',
        None,
        'Path to JAX compilation cache directory.'
    )
    flags.DEFINE_list(
        'buckets',
        ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
         '3584', '4096', '4608', '5120'],
        'Bucket sizes for caching compilations.'
    )
    flags.DEFINE_enum(
        'flash_attention_implementation',
        'triton',
        ['triton', 'cudnn', 'xla'],
        'Flash attention implementation to use.'
    )
    flags.DEFINE_integer(
        'num_diffusion_samples',
        5,
        'Number of diffusion samples to generate.',
    )

def define_alphalink_flags():
    # AlphaLink backend flags (hardcoded at the moment!)
    flags.DEFINE_string('crosslinks', None,
                        'Path to crosslink information pickle for AlphaLink.')
