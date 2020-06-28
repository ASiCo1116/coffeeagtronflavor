def add_arguments(parser):
    parser.add_argument('-s', '--subset',type = int, choices = [0, 1, 2, 3, 4], help = '0: 399 set, 1: 150 set, 2: 159 selected set(SSO), 3: 122 selected set(SS), 4:194 selected set(++SS --FS)' )
    parser.add_argument('-r', '--run_model', action = 'store_true', help = 'run model')
    parser.add_argument('-s1', '--seed1', type = int, help = 'seed of train/test', default = 1)
    parser.add_argument('-s2', '--seed2', type = int, help = 'seed of train/valid', default = 42)
    parser.add_argument('-n', '--num_of_flavor', type = int, default = 9, help = 'number of flavor')
    parser.add_argument('-s0', '--seed0', type = int, help = 'seed of 150 subset')
    parser.add_argument('-p', '--plot_f1_score', action = 'store_true', help = 'plot f1 score vs. parameters')

    return parser