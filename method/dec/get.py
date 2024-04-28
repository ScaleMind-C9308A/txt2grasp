def get_decoder(args):
    if args.dec == 'basev0':
        from .base import Basev0
        model = Basev0()
    else:
        raise ValueError(f"Model {args.dec} is not supported")

    return model