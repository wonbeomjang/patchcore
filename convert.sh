for filename in ./serve/*.onnx
do
  basename="${filename%.*}"
  trtexec --onnx="$filename" --saveEngine="$basename.trt"
#  trtexec --onnx=wide_resnet50_2.onnx --saveEngine=wide_resnet50_2.trt --verbose
done