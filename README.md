

Profiling:

original code using networkx:

  _     ._   __/__   _ _  _  _ _/_   Recorded: 12:43:42  Samples:  676
 /_//_/// /_\ / //_// / //_'/ //     Duration: 112.558   CPU time: 112.531
/   _/                      v5.0.1

Profile at /Users/poldrack/Dropbox/code/infomap_sims/infomap_sims.py:151

112.457 <module>  infomap_sims.py:1
├─ 52.944 matching_matrix_to_graph  infomap_sims.py:61
│  ├─ 38.500 Graph.add_edges_from  networkx/classes/graph.py:968
│  │     [3 frames hidden]  <built-in>
│  │        25.500 [self]  networkx/classes/graph.py
│  ├─ 8.151 [self]  infomap_sims.py
│  └─ 6.293 percentile  numpy/lib/_function_base_impl.py:4027
│        [5 frames hidden]  numpy, <built-in>
├─ 40.813 run_infomap  infomap_sims.py:92
│  ├─ 27.885 Infomap.add_networkx_graph  infomap.py:4781
│  │     [9 frames hidden]  infomap, <built-in>, networkx
│  └─ 12.929 Infomap.run  infomap.py:5110
│        [2 frames hidden]  infomap, <built-in>
└─ 17.811 create_noisy_matching_matrix  infomap_sims.py:112
   ├─ 12.092 z_to_r  infomap_sims.py:39
   └─ 4.702 [self]  infomap_sims.py


