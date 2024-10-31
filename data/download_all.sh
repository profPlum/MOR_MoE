#!/bin/bash

while (( $(ls turbulence_f1_output/*.h5 | wc -l)<4000 )); do
    papermill --execution-timeout=$(( 60*60*3 )) Interactive_Channel_Get.ipynb out_channel.ipynb
done
