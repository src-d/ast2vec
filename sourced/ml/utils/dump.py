import modelforge.dump


def dump_model(args):
    """
    Proxy for :func:`modelforge.dump.dump_model`.
    """
    if args.gcs_bucket:
        args.args = "bucket=" + args.gcs_bucket
        del args.gcs_bucket
    else:
        args.args = ""
    args.backend = None
    return modelforge.dump.dump_model(args)
