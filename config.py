import argparse

def get_config():
    config = argparse.ArgumentParser()
    config.add_argument("--save-path", default="./saved_models", type=str)

    # training
    config.add_argument("--device", default='1', type=str)
    config.add_argument("--seed", default=42, type=int)
    config.add_argument("--batch-size", default=4, type=int)
    config.add_argument("--epochs", default=2, type=int)
    config.add_argument("--showtime", default=2000, type=int)
    config.add_argument("--base-encoder-lr", default=1e-5, type=float)
    config.add_argument("--longformer-base-encoder-lr", default=1e-5, type=float)
    config.add_argument("--finetune-lr", default=1e-3, type=float)
    config.add_argument("--warm-up", default=5e-2, type=float)
    config.add_argument("--weight-decay", default=1e-5, type=float)
    config.add_argument("--early-num", default=5, type=int)
    config.add_argument("--num-tags", default=5, type=int)
    config.add_argument("--threshold", default=0.3, type=float)

    config.add_argument("--hidden-size", default=128, type=int)
    config.add_argument("--layers", default=2, type=int)
    config.add_argument("--is-bi", default=False, type=bool)
    config.add_argument("--bert-output-size", default=768, type=int)
    config.add_argument("--mlp-size", default=512, type=int)
    config.add_argument("--scale-factor", default=2, type=int)
    config.add_argument("--dropout", default=0.7, type=float)
    config.add_argument("--max-grad-norm", default=1.0, type=float)

    config.add_argument("--num-heads", default=4, type=int)
    config.add_argument("--att_dropout", default=0.1, type=float)

    config.add_argument("--mrc_dropout", type=float, default=0.7,
                        help="mrc dropout rate")
    config.add_argument("--lstm_dropout", type=float, default=0.4,
                        help="lstm dropout rate")
    config.add_argument("--classifier_act_func", type=str, default="gelu")
    config.add_argument("--classifier_intermediate_hidden_size", type=int, default=128)
    config.add_argument("--weight_start", type=float, default=1.0)
    config.add_argument("--weight_end", type=float, default=1.0)
    config.add_argument("--weight_span", type=float, default=0.1)


    config = config.parse_args()

    return config