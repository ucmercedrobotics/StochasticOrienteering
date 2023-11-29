for i in {1..50}
do
	python MCTS.py --conf config_files/config_samples_20.txt --logdir MCTS`date "+%Y_%m_%d"`/Samples/$i
done