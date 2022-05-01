echo [$(date)]: "START"
export _VERSION_=3.9
echo [$(date)]: "creating environment with python ${_VERSION_}"
conda create --prefix ./env python=${_VERSION_} -y
echo [$(date)]: "activate environment"
source activate ./env
echo [$(date)]: "install requirements"
pip install -r requirements.txt
echo [$(date)]: "install cocoapi"
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
echo [$(date)]: "create empty shell script fill"
touch OD_configure.sh
echo [$(date)]: "curl gitignore"
curl https://raw.githubusercontent.com/c17hawke/general_template/main/.gitignore > .gitignore
echo [$(date)]: "initialize git repository"
git init
echo [$(date)]: "*.zip > .gitignore"
echo "*.zip" >> .gitignore
echo [$(date)]: "END"
# to remove everything -
# rm -rf env/ .gitignore conda.yaml README.md .git/