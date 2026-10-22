#!/usr/bin/env bash
# Explicit recipe directory; requires Kaldi steps/utils/path.sh and prepared data. No training runs during verification.
# Usage: run_hybrid.sh <recipe> sat|dnn|tdnn
# env: TRI1="2000 11000" TRI2="2500 15000" SATSZ="2500 15000" NJ=6 ALI_NJ=16 NNET_JOBS=12 SEED=1
#  dnn : Beeson nnet2 recipe; SEED>1 offsets every nnet2 --srand (init, shuffling) via a patched script copy.
#  tdnn: Ribeiro et al. 2021 style nnet3 TDNN on fMLLR features (initial effective lr 0.0015), CPU.
set -euo pipefail
if [ "$#" -ne 2 ]; then printf "Usage: run_hybrid.sh <recipe> sat|dnn|tdnn\n" >&2; exit 2; fi
recipe="$1"; mode="$2"
TRI1=${TRI1:-"2000 11000"}; TRI2=${TRI2:-"2500 15000"}; SATSZ=${SATSZ:-"2500 15000"}
NJ=${NJ:-6}; ALI_NJ=${ALI_NJ:-16}; NNET_JOBS=${NNET_JOBS:-12}; SEED=${SEED:-1}
case "$mode" in sat|dnn|tdnn) ;; *) exit 2;; esac
cd "$recipe"
tag=$mode; [ "$mode" = dnn ] && [ "$SEED" != 1 ] && tag=dnn_s$SEED
exec 9>"$tag.lock"; flock -n 9 || exit 3
. ./path.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${LAVU_PYTHON:-python3}
SCORE=${LAVU_SCORE_SCRIPT:?Set LAVU_SCORE_SCRIPT to the absolute path of recipes/score_kaldi.py}
trap 'printf "failed\n" > STATUS.$tag' ERR
printf 'running\n' > STATUS.$tag

score_view() {  # score_kaldi_units.py reads <recipe>/exp/sat/decode_*
 local view=$1 src=$2
 mkdir -p $view/exp/sat; ln -sfn ../data $view/data; cp protocol.json $view/protocol.json
 for split in dev test_modal test_silent; do ln -sfn "$PWD/$src/decode_$split" $view/exp/sat/decode_$split; done
 "$PY" "$SCORE" --recipe "$PWD/$view"
}

if [ "$mode" = sat ]; then
 for split in train dev test_modal test_silent; do
  d=data/$split
  if [ ! -s "$d/cmvn.scp" ]; then
   if [ ! -s "$d/feats.scp" ]; then
    copy-feats ark,t:$d/feats.txt ark,scp:$d/feats.ark,$d/feats.scp
   fi
   utils/utt2spk_to_spk2utt.pl "$d/utt2spk" > "$d/spk2utt"
   compute-cmvn-stats --spk2utt=ark:"$d/spk2utt" scp:"$d/feats.scp" ark,scp:"$d/cmvn.ark","$d/cmvn.scp"
   utils/validate_data_dir.sh --no-wav "$d"
  fi
 done
 if [ ! -s data/lang/G.fst ]; then
  utils/prepare_lang.sh data/local/dict '<UNK>' data/local/lang data/lang
  arpa2fst --disambig-symbol='#0' --read-symbol-table=data/lang/words.txt data/local/lm.arpa data/lang/G.fst
 fi
 [ -s exp/mono/final.mdl ] || steps/train_mono.sh --nj $NJ --cmd run.pl data/train data/lang exp/mono
 [ -s exp/mono_ali/ali.1.gz ] || steps/align_si.sh --nj $NJ --cmd run.pl data/train data/lang exp/mono exp/mono_ali
 [ -s exp/tri1/final.mdl ] || steps/train_deltas.sh --cmd run.pl $TRI1 data/train data/lang exp/mono_ali exp/tri1
 [ -s exp/tri1_ali/ali.1.gz ] || steps/align_si.sh --nj $NJ --cmd run.pl data/train data/lang exp/tri1 exp/tri1_ali
 [ -s exp/tri2/final.mdl ] || steps/train_lda_mllt.sh --cmd run.pl --splice-opts '--left-context=3 --right-context=3' $TRI2 data/train data/lang exp/tri1_ali exp/tri2
 [ -s exp/tri2_ali/ali.1.gz ] || steps/align_si.sh --nj $NJ --cmd run.pl --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali
 [ -s exp/sat/final.mdl ] || steps/train_sat.sh --cmd run.pl $SATSZ data/train data/lang exp/tri2_ali exp/sat
 [ -s exp/sat/graph/HCLG.fst ] || utils/mkgraph.sh data/lang exp/sat exp/sat/graph
 for split in dev test_modal test_silent; do
  [ -s exp/sat/decode_$split/lat.1.gz ] || steps/decode_fmllr.sh --skip-scoring true --nj $NJ --cmd run.pl exp/sat/graph data/$split exp/sat/decode_$split
 done
 "$PY" "$SCORE" --recipe "$PWD"
fi

if [ "$mode" != sat ]; then
 [ -s exp/sat_ali/ali.1.gz ] || steps/align_fmllr.sh --nj $ALI_NJ --cmd run.pl data/train data/lang exp/sat exp/sat_ali
fi

if [ "$mode" = dnn ]; then
 exp=exp/nnet2; view=score_dnn; trainer=steps/nnet2/train_tanh_fast.sh
 if [ "$SEED" != 1 ]; then
  exp=exp/nnet2_s$SEED; view=score_dnn_s$SEED; trainer=local/train_tanh_fast_srand_s${SEED}.sh
  mkdir -p local
  sed 's/--srand=\$x/--srand=$((x+SRAND_OFFSET))/g' steps/nnet2/train_tanh_fast.sh > $trainer
  grep -q 'SRAND_OFFSET' $trainer
  export SRAND_OFFSET=$((SEED*1000))
 fi
 [ -s $exp/final.mdl ] || bash $trainer --cmd run.pl --stage -5 \
   --num-hidden-layers 4 --hidden-layer-dim 1024 --splice-width 4 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 --minibatch-size 128 \
   --num-jobs-nnet $NNET_JOBS --num-threads 1 --parallel-opts "" \
   --transform-dir exp/sat_ali data/train data/lang exp/sat_ali $exp
 for split in dev test_modal test_silent; do
  [ -s $exp/decode_$split/lat.1.gz ] || steps/nnet2/decode.sh --nj $NJ --cmd run.pl --skip-scoring true \
    --transform-dir exp/sat/decode_$split exp/sat/graph data/$split $exp/decode_$split
 done
 score_view $view $exp
fi

if [ "$mode" = tdnn ]; then
 for split in train dev test_modal test_silent; do
  d=data_fmllr/$split
  if [ ! -s $d/cmvn.scp ]; then
   if [ $split = train ]; then td=exp/sat_ali; nj=$ALI_NJ; else td=exp/sat/decode_$split; nj=$NJ; fi
   steps/nnet/make_fmllr_feats.sh --nj $nj --cmd run.pl --transform-dir $td $d data/$split exp/sat $d/log $d/data
   steps/compute_cmvn_stats.sh --fake $d $d/log $d/data
  fi
 done
 dir=exp/tdnn
 if [ ! -s $dir/final.mdl ]; then
  mkdir -p $dir/configs
  num_targets=$(tree-info exp/sat_ali/tree | grep num-pdfs | awk '{print $2}')
  cat > $dir/configs/network.xconfig <<XC
input dim=40 name=input
fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat
relu-batchnorm-layer name=tdnn1 dim=512
relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=512
relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=512
relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=512
relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=512
output-layer name=output dim=$num_targets max-change=1.5
XC
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
  steps/nnet3/train_dnn.py --stage=-10 --cmd=run.pl --use-gpu=false \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs=4 --trainer.optimization.num-jobs-initial=4 --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0015 --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.samples-per-iter=400000 --egs.dir="" --cleanup.remove-egs=true \
    --feat-dir=data_fmllr/train --ali-dir=exp/sat_ali --lang=data/lang --dir=$dir
 fi
 for split in dev test_modal test_silent; do
  [ -s $dir/decode_$split/lat.1.gz ] || steps/nnet3/decode.sh --nj $NJ --cmd run.pl --skip-scoring true \
    exp/sat/graph data_fmllr/$split $dir/decode_$split
 done
 score_view score_tdnn $dir
fi
printf 'complete\n' > STATUS.$tag
