utils.py:
Line 28:
alphate ——> self.alphbet
otherwise may cause following problem:
Traceback (most recent call last):
File "zhaoyuanying.py", line 84, in 
t, l = converter.encode(cpu_texts)#,scanned=False)
File "/home/mcht/tmp/crnn_pytorch/utils.py", line 84, in encode
for char in text
File "/home/mcht/tmp/crnn_pytorch/utils.py", line 84, in 
for char in text
KeyError: '-'

