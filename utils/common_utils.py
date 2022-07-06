import os
import pdb


def log(output, args):
    print(output)
    with open(os.path.join(args.log_root, args.log_filename + '.txt'), "a") as f:
        f.write(output + '\n')
