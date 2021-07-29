def mafft_run(fn_in, fn_out, op=5, verbose=False):
    import subprocess
    sh = subprocess.run(f'mafft --auto --op {op} {fn_in} > {fn_out}',
                        capture_output=True, shell=True)
    if verbose:
        print(sh.args)
        print(sh.stdout.decode())
        print(sh.stderr.decode())
    
def mafft(record_list, op=5, verbose=False):
    import tempfile, os
    from Bio import SeqIO, AlignIO
    
    f_in = tempfile.NamedTemporaryFile(delete=False)
    SeqIO.write(record_list, f_in.name, 'fasta')
    f_in.close()
    
    f_out = tempfile.NamedTemporaryFile(delete=False)
    f_out.close()
    
    mafft_run(f_in.name, f_out.name, op=op, verbose=verbose)
    
    os.unlink(f_in.name)
    
    records = AlignIO.read(f_out.name, 'fasta')
    if verbose:
        print(f_out.name)
    os.unlink(f_out.name)
    
    return(records)