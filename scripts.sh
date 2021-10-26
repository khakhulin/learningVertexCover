python run.py --cuda --num_iters 5000 --exp_name erdos_renyi_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type erdos_renyi
python run.py --cuda --num_iters 5000 --exp_name powerlaw_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type powerlaw
python run.py --cuda --num_iters 5000 --exp_name barabasi_albert_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type barabasi_albert
python run.py --cuda --num_iters 5000 --exp_name gnp_random_graph_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type gnp_random_graph
python run_test.py --exp_name erdos_renyi_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type erdos_renyi
python run_test.py --exp_name powerlaw_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type powerlaw
python run_test.py --exp_name barabasi_albert_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type barabasi_albert
python run_test.py --exp_name gnp_random_graph_min-10_max-100 --min_n 10 --max_n 100 --random_graph_type gnp_random_graph