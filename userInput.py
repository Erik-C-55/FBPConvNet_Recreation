import argparse

def getArgs(CLArgs):
    parser = argparse.ArgumentParser()
    
    # Suppress filename & add required arguments
    parser.add_argument("filename",type=str,help=argparse.SUPPRESS)
    parser.add_argument('output_dir', type=str,
                        help='directory for holding output data')
    
    # Add optional arguments
    parser.add_argument('-d','--display', action='store_true')
    parser.add_argument('-l','--lower_bound', type=int, default=5,
                        help='a random number (integer) of ellipses will be generated.  This is the lower bound for that random number.')
    parser.add_argument('-r','--res', type=int, default=512,
                        help='image resolution in pixels.  Image will be square (res x res).  Resolution should be a multiple of 16.')
    parser.add_argument('-s','--samples', type=int, default=10,
                        help='number of sample images to generate')
    parser.add_argument('-u','--upper_bound', type=int, default=15,
                        help='a random number (integer) of ellipses will be generated.  This is the upper bound for that random number.')
    
    args = parser.parse_args(CLArgs)
    
    if args.lower > args.upper:
        args.upper = args.lower + 10
    
    return args