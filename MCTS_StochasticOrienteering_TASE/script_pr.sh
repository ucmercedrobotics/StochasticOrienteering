for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
do
    grep RANDOM  config_files/config_pr.txt > config_files/result
    if [ -s config_files/result ] ; then
        sed -i '' -e '$ d' config_files/config_pr.txt
    fi
    echo "PROBABILITY_RANDOM="$i >> config_files/config_pr.txt
    #cat config_files/config_pr.txt
    python MCTS.py --conf config_files/config_time.txt --logdir MCTS`date "+%Y_%m_%d"`/Pr$i
    
done
