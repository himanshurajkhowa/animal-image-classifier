"""Run the notebooks in order, or remove notebook outputs for Git publication."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strip',action='store_true',help='Read a notebook on stdin and write a copy without outputs.')
    parser.add_argument('--start',type=int,default=1,choices=range(1,5),help='Start at this notebook number.')
    args=parser.parse_args()
    if args.strip:
        notebook=json.load(sys.stdin)
        for cell in notebook.get('cells',[]):
            if cell.get('cell_type')=='code': cell['execution_count']=None; cell['outputs']=[]
            cell.get('metadata',{}).pop('execution',None)
        json.dump(notebook,sys.stdout,ensure_ascii=False,indent=1); sys.stdout.write('\n')
    else:
        import nbformat
        from nbclient import NotebookClient
        from ipykernel.kernelspec import install
        install(prefix=str(ROOT/'.venv'),kernel_name='animal-classifier',display_name='Animal classifier Python')
        os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            if int(path.name[:2])<args.start: continue
            print('START',path.name,flush=True); started=time.time()
            notebook=nbformat.read(path,as_version=4)
            def save(**kwargs): nbformat.write(notebook,path)
            client=NotebookClient(notebook,timeout=7200,kernel_name='animal-classifier',
                                  resources={'metadata':{'path':str(ROOT/'notebooks')}},on_cell_complete=save)
            try:
                client.execute(); nbformat.write(notebook,path)
            except Exception:
                nbformat.write(notebook,path); raise
            print('DONE',path.name,'seconds',round(time.time()-started,1),flush=True)
