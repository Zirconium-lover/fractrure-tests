#!/usr/bin/env bash
# run.sh <deck.inp> <run_dir> [NAME=VALUE ...]: one torn-sheet run with the
# configuration of the disc sensitivity runs (fn0p + BARE_MASK_LAW + HINGE_CLUSTER_BIG).
# CCX_EXE: the ccx-arch2 binary (default src/ccx_2.23_pardiso, commit c75ad9b or later).
deck=$1; D=$2; shift 2
cd /home/user/ccx-arch2
S3RAD_DECK=$deck OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
CCX_EXE=${CCX_EXE:-/home/user/ccx-arch2/src/ccx_2.23_pardiso} timeout ${TLIMIT:-600000} bash test/s3rad/run_s3rad.sh "$D" \
  CCX_DAMAGE_VISCOUS_DAMPING=2e-4 CCX_DAMAGE_FACET_DEBRIS=1 CCX_DAMAGE_GRADUAL_DELETE=8 CCX_DAMAGE_DEADSOLE=1e-3 CCX_DAMAGE_DEADSOLE_LAW=1 \
  CCX_DAMAGE_BARE_MASK=0.95 CCX_DAMAGE_BARE_MASK_LAW=1 CCX_DAMAGE_HINGE=0.95 CCX_DAMAGE_HINGE_PENDANT=2 CCX_DAMAGE_HINGE_CLUSTER=1 \
  CCX_DAMAGE_HINGE_CLUSTER_BIG=1 CCX_DAMAGE_TR_STICKY=1 CCX_DAMAGE_RESCUE_CUTBACKS=1 CCX_DAMAGE_TR_MAXEVAL=100000 \
  CCX_DAMAGE_TR_MAXFACT=100000 CCX_DAMAGE_TR_MAXARM=1000 "$@" > "$D.out" 2>&1
echo "$(basename $D) rc=$? $(tail -1 $D/m.sta | awk '{print "inc",$2,"theta",$6}') del=$(grep -vc '^#' $D/m.damage) $(grep -a -h 'FRACTURE COMPLETE\] inc' $D/run.log | head -1 | cut -c1-60)" >> $(dirname $D)/summary.txt
