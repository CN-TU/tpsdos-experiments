Building Fixed-Size Spatiotemporal Models for Evolving Data Streams
===================================================================

This is the material for reproducing experiments for the paper "Building
Fixed-Size Spatiotemporal Models for Evolving Data Streams". All experiments
were performed on a Ubuntu 18.04.1 Linux system using Python 3.6.9. Python
package requirements can be installed with `pip` using

    pip3 install -r requirements.txt

Furthermore, for some experiments, `git` has to be installed.

Code for our proposed method is contained in the `tpSDOs` directory as a Python
module and is installed by issuing the above command.

Proof of Concept
----------------

1. Change to the `poc` directory.

        cd poc

2. Remove the file `results.csv`, which contains our obtained
   results.

        rm results.csv

3. Run the proof of concept implementation for several fractions of temporal
   outliers. Alternatively, you can specify the desired fraction of temporal
   outliers as script parameter.

        python3 run.py

4. Results are appended to the `results.csv` file and can be plotted using
   `python3 plot.py`.

Outlier Detection Evaluation for KDD Cup'99
-------------------------------------------

1. Change to the `outlier` directory.

        cd outlier

2. Download `kddcup.data.gz` from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
   to the `outlier` directory and extract it.

        wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz && gzip -d kddcup.data.gz

3. Perform features extraction for the KDD Cup'99 dataset, creating the
   `kddcup.npz` file.

        python3 kddcup.py

4. Run all outlier detection algorithms for KDD Cup'99.

        python3 run.py kddcup rshash swknn swrrct loda swlof tpsdose

5. Results will be appended to `results.csv`. The `results.csv` file contained
   in this archive shows our obtained results. Results for our proposed method
   are named `tpsdose`.

Outlier Detection for SWAN-SF
-----------------------------

1. Change to the directory `outlier`.

        cd outlier

2. Download the files  `partition1_instances.tar.gz`,  `partition2_instances.tar.gz`, 
   `partition3_instances.tar.gz`,  `partition4_instances.tar.gz`, 
   `partition5_instances.tar.gz` from
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM
   and place them in the `outlier` directory.

3. Extract the downloaded files to obtain directories `partition1`, `partition2`,
   `partition3`, `partition4`, `partition5` within the `outlier` directory.

        tar xf partition1_instances.tar.gz
        tar xf partition2_instances.tar.gz
        tar xf partition3_instances.tar.gz
        tar xf partition4_instances.tar.gz
        tar xf partition5_instances.tar.gz

4. Clone the https://bitbucket.org/gsudmlab/swan_features/src/master/ repository
   to `gsudmlab-swan_features`. The repository is used for feature extraction.

        git clone https://bitbucket.org/gsudmlab/swan_features/src/master/ gsudmlab-swan_features

5. Ensure you are using the version with commit hash 56eb7cb, which we used
   for our experiments.

        git -C gsudmlab-swan_features checkout 56eb7cb

6. Perform feature extraction for the SWAN-SF dataset, creating the `swan.npz`
   file.

        python3 swan.py

7. Run all outlier detection algorithms for SWAN-SF.

        python3 run.py swan rshash swknn swrrct loda swlof tpsdose

8. Results will be appended to `results.csv`. The `results.csv` file contained
   in this archive shows our obtained results. Results for our proposed method
   are named `tpsdose`.

Knowledge Discovery on Network Traffic
--------------------------------------

Due to issues related to confidentiality, security and privacy, we unfortunately
cannot make this dataset publicly available. The following steps therefore
apply to an arbitrary network capture file `capture.pcap`.

For this experiment, an installation of golang is additionally required. For
step 7, additionally an installation of `tshark` is required.

1. Install go-flows from `https://github.com/CN-TU/go-flows`.

        go get github.com/CN-TU/go-flows/...

2. Change to the `m2m` directory.

        cd m2m

3. Perform flow extraction based on feature specifications in `CAIA.json`,
   creating the file `capture.csv` from `capture.pcap`.

        go-flows run features CAIA.json export csv capture.csv source libpcap capture.pcap

4. Process flow information using the proposed algorithm, obtaining the
   file `results.pickle`.

        python3 process.py

5. Plot obtained outlier scores and the amount of sampled data points per day.

        python3 plot_scores.py
        python3 plot_sampling.py

6. For each observer, plot the magnitude spectrum of observers, 1h temporal
   plots and 24h temporal plots into the directories `fts`, `temporal_1h`
   and `temporal_24h`, respectively.

        python3 analyze.py

7. For each observer, extract a PCAP file containing network traffic
   corresponding to the respective observer into the `pcaps` directory.

        python3 extract.py

Knowledge Discovery on Darkspace Data
-------------------------------------

Please note that the used dataset has a size of ~2TB and processing of the data
takes several weeks. To only plot the results we obtained, you can use the
existing `results.pickle` file and skip to step 6.

1. Change to the `darkspace` directory.

2. Obtain the 'Patch Tuesday' dataset from
   https://www.caida.org/catalog/datasets/telescope-patch-tuesday_dataset/,
   and place the `ucsd_network_telescope.anon.*.flowtuple.cors.gz` files in
   the `darkspace` directory.

3. Obtain the legacy Corsaro software from https://github.com/CAIDA/corsaro ,
   build the `cors2ascii` tool and place it in your system's PATH.

4. Extract flow information in AGM format.

        ( for file in ucsd_network_telescope.anon.*.flowtuple.cors.gz ; do cors2ascii $file ; done ) | python3 cors2agm.py >agm.csv

5. Process flow information, obtaining the file `results.pickle`.

        python3 process.py

6. Perform frequency plots and temporal plots from `results.pickle`

        python3 plot.py

7. Plots can be found in the `fts` and `temporal_1w` directories.
