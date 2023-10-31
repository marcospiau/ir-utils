# For a simpler usage just installing pyserini from PyPI is enough, but we are going to install from source and compile
# anserini because we may be interested in using scripts inside anserini and pyserini repos.
sudo apt-get update && sudo apt-get install -y --no-install-recommends openjdk-11-jdk maven

# clean env
rm -rf pyserini anserini tools anserini-tools
BASEDIR=$PWD

echo 'Installing anserini-tools'
git clone https://github.com/castorini/anserini-tools.git && mv anserini-tools tools
cd tools && \
cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../.. && \
cd eval && cd ndeval && make && cd ../..
cd $BASEDIR

echo 'Installling pyserini'
rm -rf pyserini
git clone https://github.com/castorini/pyserini.git && cd pyserini && pip install -e .
pip install -q faiss-cpu torch
cd $BASEDIR

echo 'Building anserini'
git clone https://github.com/castorini/anserini.git && \
sudo apt install maven && mvn --version && \
cd anserini && mvn clean package appassembler:assemble -Dmaven.test.skip=true
cd $BASEDIR

echo 'Copying jars to pyserini'
mkdir -pv pyserini/pyserini/resources/jars/ && \
cp anserini/target/anserini-*-fatjar.jar pyserini/pyserini/resources/jars/