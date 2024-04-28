from .naive import Naive

def get_method(args):
    if args.method == 'naive':
        model = Naive(args)
    else:
        raise ValueError(f"Method {args.method} is not supported")
    
    return model