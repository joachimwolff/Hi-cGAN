# source this, do not execute it
#
# TF 2.21 installed via pip does not locate its own bundled CUDA libraries, and
# reports "Cannot dlopen some GPU libraries" with an empty device list. Putting
# the nvidia-* package lib directories on LD_LIBRARY_PATH fixes it.
VENV="${HICGAN_MODERN_VENV:-$HOME/venvs/hicgan-tf-modern}"
_nv=$("$VENV/bin/python" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
if [[ -n "$_nv" ]]; then
    export LD_LIBRARY_PATH="$(find "$_nv" -name lib -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH}"
fi
export PATH="$VENV/bin:$PATH"
# Keras 3 and legacy Keras 2 both work; legacy is the smaller change from the
# TF 2.15 environment, so it is the default here. Set to 0 to test Keras 3.
export TF_USE_LEGACY_KERAS="${TF_USE_LEGACY_KERAS:-1}"
unset _nv
