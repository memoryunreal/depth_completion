# depth_completion
depthmap completion/inpainting 

cleargrasp is still not working. run /cleargrasp/api/enhance_D.py

depth_refine_reconstruct function well (completion not very good)
```
pip install -r /depth_refine_reconstruct/requirements.txt
python /depth_refine_reconstruct/demo.py --hole_depth /path/to/sequence/ --output_depth /path/to/completed sequence
```
