from core import run
import random

if __name__ == '__main__':

    for i in range(1000):
        # Test only dropout
        hyper_params = {
            # 'lambda_value': random.random() * 100,
            'lambda_value': 70.29,
            # 'hidden_layer_number': random.randint(3, 7),
            'hidden_layer_number': 4,
            # 'K': random.randint(10, 100),
            'K': 39,
            # 'D_prime': random.randint(50, 100),
            'D_prime': 66,
            'hidden_unit_number': 50,
            # 'dropout_rate': random.random() * 0.5,
            'learning_rate': 1e-3,
        }

        batch_size = 128
        valid_rmse, test_rmse = run(batch_size=batch_size, **hyper_params)

        print('\t'.join(sorted(hyper_params.keys())), valid_rmse, test_rmse)
        msg = '{}\t{}\t{}'.format('\t'.join(
            str(hyper_params[key])
            for key in sorted(hyper_params.keys())), valid_rmse, test_rmse)
        print(msg)
        with open('results/ml-1m.txt', 'a') as f:
            f.write(msg + '\n')