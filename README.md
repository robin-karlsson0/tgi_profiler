# tgi_profiler
Find maximum input/output token length combinations for Text Generation Inference (TGI) deployments.


Presumes that a Hugginface user token is set as the environment variable `HF_TOKEN` and readable by
```
os.environ.get('HF_TOKEN')
```