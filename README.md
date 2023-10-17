# StochasticOrienteering
Various approaches and datasets to solve various versions of the stochastic orienteering problem

* StochasticOrienteeringMILP: solves problem using MILP approach described in https://link.springer.com/content/pdf/10.1007/978-3-642-41575-3_30.pdf 
* OrienteeringGraph: helper class modeling an instance of the orienteering problem (used by various approaches)
* MCTS_StochasticOrienteering_TASE: code for results presented in TASE submissions (extendes CASE 2022 paper)

Execute the following as such:
* StochasticOrienteeringMILP: 
1. Navigate to the proper folder -> `cd MCTS_StochasticOrienteering_TASE` 
2. `python3 -m MCTS`
You can specify the config file using `--conf`, but by default the file will be located in this directory under `config.txt`.
* MCTS_StochasticOrienteering_TASE: 
1. Navigate to the proper folder -> `cd StochasticOrienteeringMILP`
2. `python3 -m mcts_MILP` 
You can specify the config file using `--conf`, but by default the file will be located in this directory under `config.txt`.