$Env:CONDA_EXE = "/home/dmolodenskiy/AlphaPulldown/conda/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/home/dmolodenskiy/AlphaPulldown/conda"
$Env:_CONDA_EXE = "/home/dmolodenskiy/AlphaPulldown/conda/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs