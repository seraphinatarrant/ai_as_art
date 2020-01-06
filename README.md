# AI as Art
A selection of AI systems built in collaboration with artists for exhibits and performances

## Instructions
Get the repo:
`git clone https://github.com/seraphinatarrant/ai_as_art.git`
then `cd ai_as_art`

For each different project, make an environment with:
`conda create -n MYENVNAME python=3.7`

Once it is created, activate it to install everything:
`conda activate MYENVNAME`

Then navigate to the directory for the project (each will have different requirements), and do:
`pip install -r requirements.txt`

To deactivate the environment: `conda deactivate`

# Images
directory: `image_generation/`.
(Under the hood, this is a DC GAN, forked from pytorch `examples/`.)
 
## Generate
1. Use an existing config in the `config` dir, or make a new one. 
2. Activate the appropriate environment.
3. `python generate.py -c config/MYCONFIG` 

# Text
directory: `text_generation/`
TBD adding things here
