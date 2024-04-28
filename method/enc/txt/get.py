from .gpt2 import GPT2Enc

def get_txt_enc(args):
    if args.txtenc == 'base':
        pass
    elif args.txtenc == 'gpt2':
        model = GPT2Enc(block_size=args.bls, vocab_size=args.vcs, n_layer=args.nl, n_head=args.nh, n_embd=args.ne)
    else:
        raise ValueError(f"Model {args.txtenc} is not supported")
    
    return model