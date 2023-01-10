## Platform Agnostic instructions

1. To install Julia on your platform, download from the appropriate mirror and add to PATH, instructions can be found [here](https://julialang.org/downloads/platform/)

2. To set up Git on your computer follow the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html)

### Running the Experiments

Firstly we need to download this repository and setup the project, in the command terminal we type as follows:

git clone https://github.com/deveshjawla/MuZero.jl
cd MuZero.jl
julia --project -e 'import Pkg; Pkg.instantiate()'

Now using any editor of choice such as Visual Studio Code, we can start running the experiements from the dataset folder.

If you wish to use the code for other datasets then I recommend making a new folder and copying the source code from the other datasets. Now all you need to do are the following steps:

1. Data preprocessing - This step is unique for your probelem and the dataset and one should carefully analyse and decide how to process the dataset
2. Design of the Neural network - If you have a simple dataset with tabular features, then you can use an MLP as the functional model. Here then all that is left is to chose a size of the network. This will depend on your dataset.
