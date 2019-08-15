# beatai
AI Development for ECG Beat Detection
Developed within Virtual Environment under Python3.7

## Installation
```python
virtualenv -p python3 beatenv
source beatenv/bin/activate
pip install -r requirements.txt
```

### Tools

##### detection.py
```python
usage: detection.py [-h] -i INPUT [-f FILTERLOWPASS] [-a ANNOTATIONFILE]
                    [--datafile]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input db/dat file
  -f FILTERLOWPASS, --filterlowpass FILTERLOWPASS
                        Filter Low Pass Hz
  -a ANNOTATIONFILE, --annotationfile ANNOTATIONFILE
                        Annotation File
  --datafile
```


Example
1) Download mit arrhythmia data: <link>https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip</link>
2) Create data folder under root directory
3) Unzip data under data directory

```python
cd src
python detection.py --datafile -i ../data/mit/mit-bih-arrhythmia-database-1.0.0/100.dat -f 12.0 -a ../data/mit/mit-bih-arrhythmia-database-1.0.0/100.at
```

Controls:<br>
*Note: You must select the figure to focus controls.<br>
<b>space-bar:</b> Pause/Play<br>
<b>left-arrow:</b> Move One Beat Early<br>
<b>right-arrow:</b> Move One Beat Future<br>
<b>b:</b> Enter Beat Number (Must focus to Terminal) <br>
<b>Esc:</b> Exit Program



