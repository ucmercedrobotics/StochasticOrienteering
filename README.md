# StochasticOrienteering

Code tested and developed with Python 3.11.5.

Various approaches and datasets to solve various versions of the stochastic orienteering problem

* StochasticOrienteeringMILP: solves problem using the MILP approach described in https://link.springer.com/content/pdf/10.1007/978-3-642-41575-3_30.pdf.
  To run this code you need a license for Gurobi. If you do not have it, replace the line "SOLVER" in the configuration files with CBC. It will use the CBC solver (which is much slower, but free).
* OrienteeringGraph: helper class modeling an instance of the orienteering problem (used by various approaches).
* MCTS_StochasticOrienteering_TASE: code for results presented in TASE submissions (extends CASE 2022 paper).

