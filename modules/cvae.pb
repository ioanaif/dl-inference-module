
I
x_inputPlaceholder*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
A
y_inputPlaceholder*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

J
Reshape/shapeConst*%
valueB"ÿÿÿÿ      
   *
dtype0
A
ReshapeReshapey_inputReshape/shape*
T0*
Tshape0
Q
ones/shape_as_tensorConst*
dtype0*%
valueB"d         
   
7

ones/ConstConst*
valueB
 *  ?*
dtype0
I
onesFillones/shape_as_tensor
ones/Const*
T0*

index_type0
"
mulMulonesReshape*
T0
5
concat/axisConst*
value	B :*
dtype0
K
concatConcatV2x_inputmulconcat/axis*
T0*
N*

Tidx0
[
encoder/truncated_normal/shapeConst*%
valueB"            *
dtype0
J
encoder/truncated_normal/meanConst*
valueB
 *    *
dtype0
L
encoder/truncated_normal/stddevConst*
valueB
 *à"Ç=*
dtype0

(encoder/truncated_normal/TruncatedNormalTruncatedNormalencoder/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 
w
encoder/truncated_normal/mulMul(encoder/truncated_normal/TruncatedNormalencoder/truncated_normal/stddev*
T0
e
encoder/truncated_normalAddencoder/truncated_normal/mulencoder/truncated_normal/mean*
T0
l
encoder/Variable
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¤
encoder/Variable/AssignAssignencoder/Variableencoder/truncated_normal*
use_locking(*
T0*#
_class
loc:@encoder/Variable*
validate_shape(
a
encoder/Variable/readIdentityencoder/Variable*
T0*#
_class
loc:@encoder/Variable
>
encoder/zerosConst*
valueB*    *
dtype0
b
encoder/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
	container 

encoder/Variable_1/AssignAssignencoder/Variable_1encoder/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1*
validate_shape(
g
encoder/Variable_1/readIdentityencoder/Variable_1*
T0*%
_class
loc:@encoder/Variable_1
¯
encoder/Conv2DConv2Dconcatencoder/Variable/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
D
encoder/AddAddencoder/Conv2Dencoder/Variable_1/read*
T0
*
encoder/ReluReluencoder/Add*
T0
]
 encoder/truncated_normal_1/shapeConst*
dtype0*%
valueB"             
L
encoder/truncated_normal_1/meanConst*
valueB
 *    *
dtype0
N
!encoder/truncated_normal_1/stddevConst*
valueB
 *s¥=*
dtype0

*encoder/truncated_normal_1/TruncatedNormalTruncatedNormal encoder/truncated_normal_1/shape*
T0*
dtype0*
seed2 *

seed 
}
encoder/truncated_normal_1/mulMul*encoder/truncated_normal_1/TruncatedNormal!encoder/truncated_normal_1/stddev*
T0
k
encoder/truncated_normal_1Addencoder/truncated_normal_1/mulencoder/truncated_normal_1/mean*
T0
n
encoder/Variable_2
VariableV2*
dtype0*
	container *
shape: *
shared_name 
¬
encoder/Variable_2/AssignAssignencoder/Variable_2encoder/truncated_normal_1*
T0*%
_class
loc:@encoder/Variable_2*
validate_shape(*
use_locking(
g
encoder/Variable_2/readIdentityencoder/Variable_2*
T0*%
_class
loc:@encoder/Variable_2
@
encoder/zeros_1Const*
dtype0*
valueB *    
b
encoder/Variable_3
VariableV2*
dtype0*
	container *
shape: *
shared_name 
¡
encoder/Variable_3/AssignAssignencoder/Variable_3encoder/zeros_1*
use_locking(*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(
g
encoder/Variable_3/readIdentityencoder/Variable_3*
T0*%
_class
loc:@encoder/Variable_3
¹
encoder/Conv2D_1Conv2Dencoder/Reluencoder/Variable_2/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
H
encoder/Add_1Addencoder/Conv2D_1encoder/Variable_3/read*
T0
.
encoder/Relu_1Reluencoder/Add_1*
T0
Z
encoder/Flatten/flatten/ShapeConst*%
valueB"d             *
dtype0
Y
+encoder/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0
[
-encoder/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0
[
-encoder/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0
Ù
%encoder/Flatten/flatten/strided_sliceStridedSliceencoder/Flatten/flatten/Shape+encoder/Flatten/flatten/strided_slice/stack-encoder/Flatten/flatten/strided_slice/stack_1-encoder/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
Z
'encoder/Flatten/flatten/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

%encoder/Flatten/flatten/Reshape/shapePack%encoder/Flatten/flatten/strided_slice'encoder/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N
x
encoder/Flatten/flatten/ReshapeReshapeencoder/Relu_1%encoder/Flatten/flatten/Reshape/shape*
T0*
Tshape0
U
 encoder/truncated_normal_2/shapeConst*
dtype0*
valueB"      
L
encoder/truncated_normal_2/meanConst*
dtype0*
valueB
 *    
N
!encoder/truncated_normal_2/stddevConst*
valueB
 *Eñ=*
dtype0

*encoder/truncated_normal_2/TruncatedNormalTruncatedNormal encoder/truncated_normal_2/shape*

seed *
T0*
dtype0*
seed2 
}
encoder/truncated_normal_2/mulMul*encoder/truncated_normal_2/TruncatedNormal!encoder/truncated_normal_2/stddev*
T0
k
encoder/truncated_normal_2Addencoder/truncated_normal_2/mulencoder/truncated_normal_2/mean*
T0
g
encoder/Variable_4
VariableV2*
dtype0*
	container *
shape:	*
shared_name 
¬
encoder/Variable_4/AssignAssignencoder/Variable_4encoder/truncated_normal_2*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4*
validate_shape(
g
encoder/Variable_4/readIdentityencoder/Variable_4*
T0*%
_class
loc:@encoder/Variable_4
@
encoder/zeros_2Const*
valueB*    *
dtype0
b
encoder/Variable_5
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¡
encoder/Variable_5/AssignAssignencoder/Variable_5encoder/zeros_2*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_5
g
encoder/Variable_5/readIdentityencoder/Variable_5*
T0*%
_class
loc:@encoder/Variable_5

encoder/MatMulMatMulencoder/Flatten/flatten/Reshapeencoder/Variable_4/read*
T0*
transpose_a( *
transpose_b( 
F
encoder/Add_2Addencoder/MatMulencoder/Variable_5/read*
T0
U
 encoder/truncated_normal_3/shapeConst*
valueB"      *
dtype0
L
encoder/truncated_normal_3/meanConst*
valueB
 *    *
dtype0
N
!encoder/truncated_normal_3/stddevConst*
valueB
 *Eñ=*
dtype0

*encoder/truncated_normal_3/TruncatedNormalTruncatedNormal encoder/truncated_normal_3/shape*
T0*
dtype0*
seed2 *

seed 
}
encoder/truncated_normal_3/mulMul*encoder/truncated_normal_3/TruncatedNormal!encoder/truncated_normal_3/stddev*
T0
k
encoder/truncated_normal_3Addencoder/truncated_normal_3/mulencoder/truncated_normal_3/mean*
T0
g
encoder/Variable_6
VariableV2*
shared_name *
dtype0*
	container *
shape:	
¬
encoder/Variable_6/AssignAssignencoder/Variable_6encoder/truncated_normal_3*
use_locking(*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(
g
encoder/Variable_6/readIdentityencoder/Variable_6*
T0*%
_class
loc:@encoder/Variable_6
@
encoder/zeros_3Const*
valueB*    *
dtype0
b
encoder/Variable_7
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¡
encoder/Variable_7/AssignAssignencoder/Variable_7encoder/zeros_3*
use_locking(*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(
g
encoder/Variable_7/readIdentityencoder/Variable_7*
T0*%
_class
loc:@encoder/Variable_7

encoder/MatMul_1MatMulencoder/Flatten/flatten/Reshapeencoder/Variable_6/read*
T0*
transpose_a( *
transpose_b( 
H
encoder/Add_3Addencoder/MatMul_1encoder/Variable_7/read*
T0
H
random_normal/shapeConst*
valueB"d      *
dtype0
?
random_normal/meanConst*
valueB
 *    *
dtype0
A
random_normal/stddevConst*
valueB
 *  ?*
dtype0
~
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0
[
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0
D
random_normalAddrandom_normal/mulrandom_normal/mean*
T0
8
z_inputPlaceholder*
dtype0*
shape
:d
"
ExpExpencoder/Add_3*
T0

SqrtSqrtExp*
T0
*
Mul_1MulSqrtrandom_normal*
T0
)
AddAddencoder/Add_2Mul_1*
T0
7
concat_1/axisConst*
value	B :*
dtype0
O
concat_1ConcatV2Addy_inputconcat_1/axis*
N*

Tidx0*
T0
S
decoder/truncated_normal/shapeConst*
dtype0*
valueB"     
J
decoder/truncated_normal/meanConst*
dtype0*
valueB
 *    
L
decoder/truncated_normal/stddevConst*
valueB
 *²Rî>*
dtype0

(decoder/truncated_normal/TruncatedNormalTruncatedNormaldecoder/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
w
decoder/truncated_normal/mulMul(decoder/truncated_normal/TruncatedNormaldecoder/truncated_normal/stddev*
T0
e
decoder/truncated_normalAdddecoder/truncated_normal/muldecoder/truncated_normal/mean*
T0
e
decoder/Variable
VariableV2*
dtype0*
	container *
shape:	*
shared_name 
¤
decoder/Variable/AssignAssigndecoder/Variabledecoder/truncated_normal*
T0*#
_class
loc:@decoder/Variable*
validate_shape(*
use_locking(
a
decoder/Variable/readIdentitydecoder/Variable*
T0*#
_class
loc:@decoder/Variable
?
decoder/zerosConst*
valueB*    *
dtype0
c
decoder/Variable_1
VariableV2*
dtype0*
	container *
shape:*
shared_name 

decoder/Variable_1/AssignAssigndecoder/Variable_1decoder/zeros*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(*
use_locking(
g
decoder/Variable_1/readIdentitydecoder/Variable_1*
T0*%
_class
loc:@decoder/Variable_1
h
decoder/MatMulMatMulconcat_1decoder/Variable/read*
transpose_a( *
transpose_b( *
T0
D
decoder/AddAdddecoder/MatMuldecoder/Variable_1/read*
T0
*
decoder/ReluReludecoder/Add*
T0
R
decoder/Reshape/shapeConst*%
valueB"ÿÿÿÿ         *
dtype0
V
decoder/ReshapeReshapedecoder/Reludecoder/Reshape/shape*
T0*
Tshape0
]
 decoder/truncated_normal_1/shapeConst*%
valueB"            *
dtype0
L
decoder/truncated_normal_1/meanConst*
valueB
 *    *
dtype0
N
!decoder/truncated_normal_1/stddevConst*
valueB
 *é=*
dtype0

*decoder/truncated_normal_1/TruncatedNormalTruncatedNormal decoder/truncated_normal_1/shape*

seed *
T0*
dtype0*
seed2 
}
decoder/truncated_normal_1/mulMul*decoder/truncated_normal_1/TruncatedNormal!decoder/truncated_normal_1/stddev*
T0
k
decoder/truncated_normal_1Adddecoder/truncated_normal_1/muldecoder/truncated_normal_1/mean*
T0
n
decoder/Variable_2
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¬
decoder/Variable_2/AssignAssigndecoder/Variable_2decoder/truncated_normal_1*
T0*%
_class
loc:@decoder/Variable_2*
validate_shape(*
use_locking(
g
decoder/Variable_2/readIdentitydecoder/Variable_2*
T0*%
_class
loc:@decoder/Variable_2
@
decoder/zeros_1Const*
valueB*    *
dtype0
b
decoder/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
	container 
¡
decoder/Variable_3/AssignAssigndecoder/Variable_3decoder/zeros_1*
use_locking(*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(
g
decoder/Variable_3/readIdentitydecoder/Variable_3*
T0*%
_class
loc:@decoder/Variable_3
b
%decoder/conv2d_transpose/output_shapeConst*%
valueB"d            *
dtype0
÷
decoder/conv2d_transposeConv2DBackpropInput%decoder/conv2d_transpose/output_shapedecoder/Variable_2/readdecoder/Reshape*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
P
decoder/Add_1Adddecoder/conv2d_transposedecoder/Variable_3/read*
T0
.
decoder/Relu_1Reludecoder/Add_1*
T0
]
 decoder/truncated_normal_2/shapeConst*
dtype0*%
valueB"            
L
decoder/truncated_normal_2/meanConst*
valueB
 *    *
dtype0
N
!decoder/truncated_normal_2/stddevConst*
valueB
 *s¥>*
dtype0

*decoder/truncated_normal_2/TruncatedNormalTruncatedNormal decoder/truncated_normal_2/shape*

seed *
T0*
dtype0*
seed2 
}
decoder/truncated_normal_2/mulMul*decoder/truncated_normal_2/TruncatedNormal!decoder/truncated_normal_2/stddev*
T0
k
decoder/truncated_normal_2Adddecoder/truncated_normal_2/muldecoder/truncated_normal_2/mean*
T0
n
decoder/Variable_4
VariableV2*
shared_name *
dtype0*
	container *
shape:
¬
decoder/Variable_4/AssignAssigndecoder/Variable_4decoder/truncated_normal_2*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4*
validate_shape(
g
decoder/Variable_4/readIdentitydecoder/Variable_4*
T0*%
_class
loc:@decoder/Variable_4
@
decoder/zeros_2Const*
dtype0*
valueB*    
b
decoder/Variable_5
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¡
decoder/Variable_5/AssignAssigndecoder/Variable_5decoder/zeros_2*
use_locking(*
T0*%
_class
loc:@decoder/Variable_5*
validate_shape(
g
decoder/Variable_5/readIdentitydecoder/Variable_5*
T0*%
_class
loc:@decoder/Variable_5
d
'decoder/conv2d_transpose_1/output_shapeConst*%
valueB"d            *
dtype0
ú
decoder/conv2d_transpose_1Conv2DBackpropInput'decoder/conv2d_transpose_1/output_shapedecoder/Variable_4/readdecoder/Relu_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
R
decoder/Add_2Adddecoder/conv2d_transpose_1decoder/Variable_5/read*
T0
.
decoder/Relu_2Reludecoder/Add_2*
T0
3
decoder/SigmoidSigmoiddecoder/Relu_2*
T0
C
decoder/x_decoder_mean_outputIdentitydecoder/Sigmoid*
T0
7

loss/add/xConst*
dtype0*
valueB
 *ÿæÛ.
C
loss/addAdd
loss/add/xdecoder/x_decoder_mean_output*
T0
"
loss/LogLogloss/add*
T0
+
loss/mulMulx_inputloss/Log*
T0
7

loss/sub/xConst*
valueB
 *  ?*
dtype0
-
loss/subSub
loss/sub/xx_input*
T0
9
loss/sub_1/xConst*
dtype0*
valueB
 *  ?
G

loss/sub_1Subloss/sub_1/xdecoder/x_decoder_mean_output*
T0
&

loss/Log_1Log
loss/sub_1*
T0
0

loss/mul_1Mulloss/sub
loss/Log_1*
T0
0

loss/add_1Addloss/mul
loss/mul_1*
T0
D
loss/Sum/reduction_indicesConst*
value	B :*
dtype0
]
loss/SumSum
loss/add_1loss/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
"
loss/NegNegloss/Sum*
T0
C

loss/ConstConst*
dtype0*!
valueB"          
M
	loss/MeanMeanloss/Neg
loss/Const*

Tidx0*
	keep_dims( *
T0
I
loss/clip_by_value/Minimum/yConst*
valueB
 *  ÈB*
dtype0
[
loss/clip_by_value/MinimumMinimumencoder/Add_3loss/clip_by_value/Minimum/y*
T0
A
loss/clip_by_value/yConst*
dtype0*
valueB
 *ÿæÛ®
X
loss/clip_by_valueMaximumloss/clip_by_value/Minimumloss/clip_by_value/y*
T0
9
loss/add_2/xConst*
valueB
 *  ?*
dtype0
<

loss/add_2Addloss/add_2/xloss/clip_by_value*
T0
-
loss/SquareSquareencoder/Add_2*
T0
3

loss/sub_2Sub
loss/add_2loss/Square*
T0
,
loss/ExpExploss/clip_by_value*
T0
0

loss/sub_3Sub
loss/sub_2loss/Exp*
T0
F
loss/Sum_1/reduction_indicesConst*
dtype0*
value	B :
a

loss/Sum_1Sum
loss/sub_3loss/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0
9
loss/mul_2/xConst*
dtype0*
valueB
 *   ¿
4

loss/mul_2Mulloss/mul_2/x
loss/Sum_1*
T0
:
loss/Const_1Const*
valueB: *
dtype0
S
loss/Mean_1Mean
loss/mul_2loss/Const_1*

Tidx0*
	keep_dims( *
T0
'

loss/IsNanIsNan	loss/Mean*
T0
:
loss/Select/tConst*
dtype0*
valueB
 *    
D
loss/SelectSelect
loss/IsNanloss/Select/t	loss/Mean*
T0
+
loss/IsNan_1IsNanloss/Mean_1*
T0
<
loss/Select_1/tConst*
dtype0*
valueB
 *    
L
loss/Select_1Selectloss/IsNan_1loss/Select_1/tloss/Mean_1*
T0
6

loss/add_3Addloss/Selectloss/Select_1*
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
R
%gradients/loss/Select_grad/zeros_likeConst*
valueB
 *    *
dtype0
w
!gradients/loss/Select_grad/SelectSelect
loss/IsNangradients/Fill%gradients/loss/Select_grad/zeros_like*
T0
y
#gradients/loss/Select_grad/Select_1Select
loss/IsNan%gradients/loss/Select_grad/zeros_likegradients/Fill*
T0
T
'gradients/loss/Select_1_grad/zeros_likeConst*
valueB
 *    *
dtype0
}
#gradients/loss/Select_1_grad/SelectSelectloss/IsNan_1gradients/Fill'gradients/loss/Select_1_grad/zeros_like*
T0

%gradients/loss/Select_1_grad/Select_1Selectloss/IsNan_1'gradients/loss/Select_1_grad/zeros_likegradients/Fill*
T0
_
&gradients/loss/Mean_grad/Reshape/shapeConst*!
valueB"         *
dtype0

 gradients/loss/Mean_grad/ReshapeReshape#gradients/loss/Select_grad/Select_1&gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0
W
gradients/loss/Mean_grad/ConstConst*!
valueB"d         *
dtype0

gradients/loss/Mean_grad/TileTile gradients/loss/Mean_grad/Reshapegradients/loss/Mean_grad/Const*

Tmultiples0*
T0
M
 gradients/loss/Mean_grad/Const_1Const*
valueB
 *  /E*
dtype0
u
 gradients/loss/Mean_grad/truedivRealDivgradients/loss/Mean_grad/Tile gradients/loss/Mean_grad/Const_1*
T0
V
(gradients/loss/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0

"gradients/loss/Mean_1_grad/ReshapeReshape%gradients/loss/Select_1_grad/Select_1(gradients/loss/Mean_1_grad/Reshape/shape*
T0*
Tshape0
N
 gradients/loss/Mean_1_grad/ConstConst*
valueB:d*
dtype0

gradients/loss/Mean_1_grad/TileTile"gradients/loss/Mean_1_grad/Reshape gradients/loss/Mean_1_grad/Const*
T0*

Tmultiples0
O
"gradients/loss/Mean_1_grad/Const_1Const*
valueB
 *  ÈB*
dtype0
{
"gradients/loss/Mean_1_grad/truedivRealDivgradients/loss/Mean_1_grad/Tile"gradients/loss/Mean_1_grad/Const_1*
T0
M
gradients/loss/Neg_grad/NegNeg gradients/loss/Mean_grad/truediv*
T0
H
gradients/loss/mul_2_grad/ShapeConst*
valueB *
dtype0
O
!gradients/loss/mul_2_grad/Shape_1Const*
valueB:d*
dtype0

/gradients/loss/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/mul_2_grad/Shape!gradients/loss/mul_2_grad/Shape_1*
T0
]
gradients/loss/mul_2_grad/MulMul"gradients/loss/Mean_1_grad/truediv
loss/Sum_1*
T0

gradients/loss/mul_2_grad/SumSumgradients/loss/mul_2_grad/Mul/gradients/loss/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

!gradients/loss/mul_2_grad/ReshapeReshapegradients/loss/mul_2_grad/Sumgradients/loss/mul_2_grad/Shape*
T0*
Tshape0
a
gradients/loss/mul_2_grad/Mul_1Mulloss/mul_2/x"gradients/loss/Mean_1_grad/truediv*
T0
 
gradients/loss/mul_2_grad/Sum_1Sumgradients/loss/mul_2_grad/Mul_11gradients/loss/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

#gradients/loss/mul_2_grad/Reshape_1Reshapegradients/loss/mul_2_grad/Sum_1!gradients/loss/mul_2_grad/Shape_1*
T0*
Tshape0
Z
gradients/loss/Sum_grad/ShapeConst*%
valueB"d            *
dtype0
x
gradients/loss/Sum_grad/SizeConst*
value	B :*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0

gradients/loss/Sum_grad/addAddloss/Sum/reduction_indicesgradients/loss/Sum_grad/Size*
T0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape

gradients/loss/Sum_grad/modFloorModgradients/loss/Sum_grad/addgradients/loss/Sum_grad/Size*
T0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape
z
gradients/loss/Sum_grad/Shape_1Const*
valueB *0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0

#gradients/loss/Sum_grad/range/startConst*
value	B : *0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0

#gradients/loss/Sum_grad/range/deltaConst*
value	B :*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0
Ì
gradients/loss/Sum_grad/rangeRange#gradients/loss/Sum_grad/range/startgradients/loss/Sum_grad/Size#gradients/loss/Sum_grad/range/delta*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*

Tidx0
~
"gradients/loss/Sum_grad/Fill/valueConst*
value	B :*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0
¶
gradients/loss/Sum_grad/FillFillgradients/loss/Sum_grad/Shape_1"gradients/loss/Sum_grad/Fill/value*
T0*

index_type0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape
ó
%gradients/loss/Sum_grad/DynamicStitchDynamicStitchgradients/loss/Sum_grad/rangegradients/loss/Sum_grad/modgradients/loss/Sum_grad/Shapegradients/loss/Sum_grad/Fill*
T0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
N
}
!gradients/loss/Sum_grad/Maximum/yConst*
value	B :*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape*
dtype0
¯
gradients/loss/Sum_grad/MaximumMaximum%gradients/loss/Sum_grad/DynamicStitch!gradients/loss/Sum_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape
§
 gradients/loss/Sum_grad/floordivFloorDivgradients/loss/Sum_grad/Shapegradients/loss/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients/loss/Sum_grad/Shape

gradients/loss/Sum_grad/ReshapeReshapegradients/loss/Neg_grad/Neg%gradients/loss/Sum_grad/DynamicStitch*
T0*
Tshape0

gradients/loss/Sum_grad/TileTilegradients/loss/Sum_grad/Reshape gradients/loss/Sum_grad/floordiv*

Tmultiples0*
T0
T
gradients/loss/Sum_1_grad/ShapeConst*
valueB"d      *
dtype0
|
gradients/loss/Sum_1_grad/SizeConst*
dtype0*
value	B :*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape

gradients/loss/Sum_1_grad/addAddloss/Sum_1/reduction_indicesgradients/loss/Sum_1_grad/Size*
T0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape
¥
gradients/loss/Sum_1_grad/modFloorModgradients/loss/Sum_1_grad/addgradients/loss/Sum_1_grad/Size*
T0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape
~
!gradients/loss/Sum_1_grad/Shape_1Const*
valueB *2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape*
dtype0

%gradients/loss/Sum_1_grad/range/startConst*
dtype0*
value	B : *2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape

%gradients/loss/Sum_1_grad/range/deltaConst*
value	B :*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape*
dtype0
Ö
gradients/loss/Sum_1_grad/rangeRange%gradients/loss/Sum_1_grad/range/startgradients/loss/Sum_1_grad/Size%gradients/loss/Sum_1_grad/range/delta*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape*

Tidx0

$gradients/loss/Sum_1_grad/Fill/valueConst*
value	B :*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape*
dtype0
¾
gradients/loss/Sum_1_grad/FillFill!gradients/loss/Sum_1_grad/Shape_1$gradients/loss/Sum_1_grad/Fill/value*
T0*

index_type0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape
ÿ
'gradients/loss/Sum_1_grad/DynamicStitchDynamicStitchgradients/loss/Sum_1_grad/rangegradients/loss/Sum_1_grad/modgradients/loss/Sum_1_grad/Shapegradients/loss/Sum_1_grad/Fill*
N*
T0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape

#gradients/loss/Sum_1_grad/Maximum/yConst*
value	B :*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape*
dtype0
·
!gradients/loss/Sum_1_grad/MaximumMaximum'gradients/loss/Sum_1_grad/DynamicStitch#gradients/loss/Sum_1_grad/Maximum/y*
T0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape
¯
"gradients/loss/Sum_1_grad/floordivFloorDivgradients/loss/Sum_1_grad/Shape!gradients/loss/Sum_1_grad/Maximum*
T0*2
_class(
&$loc:@gradients/loss/Sum_1_grad/Shape

!gradients/loss/Sum_1_grad/ReshapeReshape#gradients/loss/mul_2_grad/Reshape_1'gradients/loss/Sum_1_grad/DynamicStitch*
T0*
Tshape0

gradients/loss/Sum_1_grad/TileTile!gradients/loss/Sum_1_grad/Reshape"gradients/loss/Sum_1_grad/floordiv*

Tmultiples0*
T0
M
gradients/loss/sub_3_grad/NegNeggradients/loss/Sum_1_grad/Tile*
T0
H
gradients/loss/mul_grad/ShapeShapex_input*
T0*
out_type0
\
gradients/loss/mul_grad/Shape_1Const*%
valueB"d            *
dtype0

-gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/mul_grad/Shapegradients/loss/mul_grad/Shape_1*
T0
S
gradients/loss/mul_grad/MulMulgradients/loss/Sum_grad/Tileloss/Log*
T0

gradients/loss/mul_grad/SumSumgradients/loss/mul_grad/Mul-gradients/loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
}
gradients/loss/mul_grad/ReshapeReshapegradients/loss/mul_grad/Sumgradients/loss/mul_grad/Shape*
T0*
Tshape0
T
gradients/loss/mul_grad/Mul_1Mulx_inputgradients/loss/Sum_grad/Tile*
T0

gradients/loss/mul_grad/Sum_1Sumgradients/loss/mul_grad/Mul_1/gradients/loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

!gradients/loss/mul_grad/Reshape_1Reshapegradients/loss/mul_grad/Sum_1gradients/loss/mul_grad/Shape_1*
T0*
Tshape0
K
gradients/loss/mul_1_grad/ShapeShapeloss/sub*
T0*
out_type0
^
!gradients/loss/mul_1_grad/Shape_1Const*%
valueB"d            *
dtype0

/gradients/loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/mul_1_grad/Shape!gradients/loss/mul_1_grad/Shape_1*
T0
W
gradients/loss/mul_1_grad/MulMulgradients/loss/Sum_grad/Tile
loss/Log_1*
T0

gradients/loss/mul_1_grad/SumSumgradients/loss/mul_1_grad/Mul/gradients/loss/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

!gradients/loss/mul_1_grad/ReshapeReshapegradients/loss/mul_1_grad/Sumgradients/loss/mul_1_grad/Shape*
T0*
Tshape0
W
gradients/loss/mul_1_grad/Mul_1Mulloss/subgradients/loss/Sum_grad/Tile*
T0
 
gradients/loss/mul_1_grad/Sum_1Sumgradients/loss/mul_1_grad/Mul_11gradients/loss/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

#gradients/loss/mul_1_grad/Reshape_1Reshapegradients/loss/mul_1_grad/Sum_1!gradients/loss/mul_1_grad/Shape_1*
T0*
Tshape0
M
gradients/loss/sub_2_grad/NegNeggradients/loss/Sum_1_grad/Tile*
T0
T
gradients/loss/Exp_grad/mulMulgradients/loss/sub_3_grad/Negloss/Exp*
T0
g
"gradients/loss/Log_grad/Reciprocal
Reciprocalloss/add"^gradients/loss/mul_grad/Reshape_1*
T0
r
gradients/loss/Log_grad/mulMul!gradients/loss/mul_grad/Reshape_1"gradients/loss/Log_grad/Reciprocal*
T0
m
$gradients/loss/Log_1_grad/Reciprocal
Reciprocal
loss/sub_1$^gradients/loss/mul_1_grad/Reshape_1*
T0
x
gradients/loss/Log_1_grad/mulMul#gradients/loss/mul_1_grad/Reshape_1$gradients/loss/Log_1_grad/Reciprocal*
T0
H
gradients/loss/add_2_grad/ShapeConst*
valueB *
dtype0
V
!gradients/loss/add_2_grad/Shape_1Const*
valueB"d      *
dtype0

/gradients/loss/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/add_2_grad/Shape!gradients/loss/add_2_grad/Shape_1*
T0

gradients/loss/add_2_grad/SumSumgradients/loss/Sum_1_grad/Tile/gradients/loss/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

!gradients/loss/add_2_grad/ReshapeReshapegradients/loss/add_2_grad/Sumgradients/loss/add_2_grad/Shape*
T0*
Tshape0

gradients/loss/add_2_grad/Sum_1Sumgradients/loss/Sum_1_grad/Tile1gradients/loss/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

#gradients/loss/add_2_grad/Reshape_1Reshapegradients/loss/add_2_grad/Sum_1!gradients/loss/add_2_grad/Shape_1*
T0*
Tshape0
m
 gradients/loss/Square_grad/ConstConst^gradients/loss/sub_2_grad/Neg*
dtype0*
valueB
 *   @
_
gradients/loss/Square_grad/MulMulencoder/Add_2 gradients/loss/Square_grad/Const*
T0
o
 gradients/loss/Square_grad/Mul_1Mulgradients/loss/sub_2_grad/Neggradients/loss/Square_grad/Mul*
T0
F
gradients/loss/add_grad/ShapeConst*
dtype0*
valueB 
\
gradients/loss/add_grad/Shape_1Const*%
valueB"d            *
dtype0

-gradients/loss/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/add_grad/Shapegradients/loss/add_grad/Shape_1*
T0

gradients/loss/add_grad/SumSumgradients/loss/Log_grad/mul-gradients/loss/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
}
gradients/loss/add_grad/ReshapeReshapegradients/loss/add_grad/Sumgradients/loss/add_grad/Shape*
T0*
Tshape0

gradients/loss/add_grad/Sum_1Sumgradients/loss/Log_grad/mul/gradients/loss/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

!gradients/loss/add_grad/Reshape_1Reshapegradients/loss/add_grad/Sum_1gradients/loss/add_grad/Shape_1*
T0*
Tshape0
H
gradients/loss/sub_1_grad/ShapeConst*
valueB *
dtype0
^
!gradients/loss/sub_1_grad/Shape_1Const*
dtype0*%
valueB"d            

/gradients/loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/sub_1_grad/Shape!gradients/loss/sub_1_grad/Shape_1*
T0

gradients/loss/sub_1_grad/SumSumgradients/loss/Log_1_grad/mul/gradients/loss/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

!gradients/loss/sub_1_grad/ReshapeReshapegradients/loss/sub_1_grad/Sumgradients/loss/sub_1_grad/Shape*
T0*
Tshape0

gradients/loss/sub_1_grad/Sum_1Sumgradients/loss/Log_1_grad/mul1gradients/loss/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
N
gradients/loss/sub_1_grad/NegNeggradients/loss/sub_1_grad/Sum_1*
T0

#gradients/loss/sub_1_grad/Reshape_1Reshapegradients/loss/sub_1_grad/Neg!gradients/loss/sub_1_grad/Shape_1*
T0*
Tshape0

gradients/AddNAddNgradients/loss/Exp_grad/mul#gradients/loss/add_2_grad/Reshape_1*
N*
T0*.
_class$
" loc:@gradients/loss/Exp_grad/mul
\
'gradients/loss/clip_by_value_grad/ShapeConst*
dtype0*
valueB"d      
R
)gradients/loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0
^
)gradients/loss/clip_by_value_grad/Shape_2Const*
valueB"d      *
dtype0
Z
-gradients/loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0
¤
'gradients/loss/clip_by_value_grad/zerosFill)gradients/loss/clip_by_value_grad/Shape_2-gradients/loss/clip_by_value_grad/zeros/Const*
T0*

index_type0
y
.gradients/loss/clip_by_value_grad/GreaterEqualGreaterEqualloss/clip_by_value/Minimumloss/clip_by_value/y*
T0
­
7gradients/loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/loss/clip_by_value_grad/Shape)gradients/loss/clip_by_value_grad/Shape_1*
T0
¤
(gradients/loss/clip_by_value_grad/SelectSelect.gradients/loss/clip_by_value_grad/GreaterEqualgradients/AddN'gradients/loss/clip_by_value_grad/zeros*
T0
¦
*gradients/loss/clip_by_value_grad/Select_1Select.gradients/loss/clip_by_value_grad/GreaterEqual'gradients/loss/clip_by_value_grad/zerosgradients/AddN*
T0
µ
%gradients/loss/clip_by_value_grad/SumSum(gradients/loss/clip_by_value_grad/Select7gradients/loss/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

)gradients/loss/clip_by_value_grad/ReshapeReshape%gradients/loss/clip_by_value_grad/Sum'gradients/loss/clip_by_value_grad/Shape*
T0*
Tshape0
»
'gradients/loss/clip_by_value_grad/Sum_1Sum*gradients/loss/clip_by_value_grad/Select_19gradients/loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
¡
+gradients/loss/clip_by_value_grad/Reshape_1Reshape'gradients/loss/clip_by_value_grad/Sum_1)gradients/loss/clip_by_value_grad/Shape_1*
T0*
Tshape0
¨
gradients/AddN_1AddN!gradients/loss/add_grad/Reshape_1#gradients/loss/sub_1_grad/Reshape_1*
T0*4
_class*
(&loc:@gradients/loss/add_grad/Reshape_1*
N
d
/gradients/loss/clip_by_value/Minimum_grad/ShapeConst*
valueB"d      *
dtype0
Z
1gradients/loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0
f
1gradients/loss/clip_by_value/Minimum_grad/Shape_2Const*
valueB"d      *
dtype0
b
5gradients/loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
¼
/gradients/loss/clip_by_value/Minimum_grad/zerosFill1gradients/loss/clip_by_value/Minimum_grad/Shape_25gradients/loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0
v
3gradients/loss/clip_by_value/Minimum_grad/LessEqual	LessEqualencoder/Add_3loss/clip_by_value/Minimum/y*
T0
Å
?gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/loss/clip_by_value/Minimum_grad/Shape1gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0
Ô
0gradients/loss/clip_by_value/Minimum_grad/SelectSelect3gradients/loss/clip_by_value/Minimum_grad/LessEqual)gradients/loss/clip_by_value_grad/Reshape/gradients/loss/clip_by_value/Minimum_grad/zeros*
T0
Ö
2gradients/loss/clip_by_value/Minimum_grad/Select_1Select3gradients/loss/clip_by_value/Minimum_grad/LessEqual/gradients/loss/clip_by_value/Minimum_grad/zeros)gradients/loss/clip_by_value_grad/Reshape*
T0
Í
-gradients/loss/clip_by_value/Minimum_grad/SumSum0gradients/loss/clip_by_value/Minimum_grad/Select?gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
³
1gradients/loss/clip_by_value/Minimum_grad/ReshapeReshape-gradients/loss/clip_by_value/Minimum_grad/Sum/gradients/loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0
Ó
/gradients/loss/clip_by_value/Minimum_grad/Sum_1Sum2gradients/loss/clip_by_value/Minimum_grad/Select_1Agradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
¹
3gradients/loss/clip_by_value/Minimum_grad/Reshape_1Reshape/gradients/loss/clip_by_value/Minimum_grad/Sum_11gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0
e
*gradients/decoder/Sigmoid_grad/SigmoidGradSigmoidGraddecoder/Sigmoidgradients/AddN_1*
T0
w
&gradients/decoder/Relu_2_grad/ReluGradReluGrad*gradients/decoder/Sigmoid_grad/SigmoidGraddecoder/Relu_2*
T0
_
"gradients/decoder/Add_2_grad/ShapeConst*%
valueB"d            *
dtype0
R
$gradients/decoder/Add_2_grad/Shape_1Const*
valueB:*
dtype0

2gradients/decoder/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/decoder/Add_2_grad/Shape$gradients/decoder/Add_2_grad/Shape_1*
T0
©
 gradients/decoder/Add_2_grad/SumSum&gradients/decoder/Relu_2_grad/ReluGrad2gradients/decoder/Add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

$gradients/decoder/Add_2_grad/ReshapeReshape gradients/decoder/Add_2_grad/Sum"gradients/decoder/Add_2_grad/Shape*
T0*
Tshape0
­
"gradients/decoder/Add_2_grad/Sum_1Sum&gradients/decoder/Relu_2_grad/ReluGrad4gradients/decoder/Add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

&gradients/decoder/Add_2_grad/Reshape_1Reshape"gradients/decoder/Add_2_grad/Sum_1$gradients/decoder/Add_2_grad/Shape_1*
T0*
Tshape0
l
/gradients/decoder/conv2d_transpose_1_grad/ShapeConst*%
valueB"            *
dtype0
´
>gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilterConv2DBackpropFilter$gradients/decoder/Add_2_grad/Reshape/gradients/decoder/conv2d_transpose_1_grad/Shapedecoder/Relu_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
ð
0gradients/decoder/conv2d_transpose_1_grad/Conv2DConv2D$gradients/decoder/Add_2_grad/Reshapedecoder/Variable_4/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
}
&gradients/decoder/Relu_1_grad/ReluGradReluGrad0gradients/decoder/conv2d_transpose_1_grad/Conv2Ddecoder/Relu_1*
T0
_
"gradients/decoder/Add_1_grad/ShapeConst*
dtype0*%
valueB"d            
R
$gradients/decoder/Add_1_grad/Shape_1Const*
valueB:*
dtype0

2gradients/decoder/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/decoder/Add_1_grad/Shape$gradients/decoder/Add_1_grad/Shape_1*
T0
©
 gradients/decoder/Add_1_grad/SumSum&gradients/decoder/Relu_1_grad/ReluGrad2gradients/decoder/Add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

$gradients/decoder/Add_1_grad/ReshapeReshape gradients/decoder/Add_1_grad/Sum"gradients/decoder/Add_1_grad/Shape*
T0*
Tshape0
­
"gradients/decoder/Add_1_grad/Sum_1Sum&gradients/decoder/Relu_1_grad/ReluGrad4gradients/decoder/Add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

&gradients/decoder/Add_1_grad/Reshape_1Reshape"gradients/decoder/Add_1_grad/Sum_1$gradients/decoder/Add_1_grad/Shape_1*
T0*
Tshape0
j
-gradients/decoder/conv2d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
±
<gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter$gradients/decoder/Add_1_grad/Reshape-gradients/decoder/conv2d_transpose_grad/Shapedecoder/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
î
.gradients/decoder/conv2d_transpose_grad/Conv2DConv2D$gradients/decoder/Add_1_grad/Reshapedecoder/Variable_2/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Y
$gradients/decoder/Reshape_grad/ShapeConst*
valueB"d     *
dtype0

&gradients/decoder/Reshape_grad/ReshapeReshape.gradients/decoder/conv2d_transpose_grad/Conv2D$gradients/decoder/Reshape_grad/Shape*
T0*
Tshape0
o
$gradients/decoder/Relu_grad/ReluGradReluGrad&gradients/decoder/Reshape_grad/Reshapedecoder/Relu*
T0
U
 gradients/decoder/Add_grad/ShapeConst*
dtype0*
valueB"d     
Q
"gradients/decoder/Add_grad/Shape_1Const*
valueB:*
dtype0

0gradients/decoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/decoder/Add_grad/Shape"gradients/decoder/Add_grad/Shape_1*
T0
£
gradients/decoder/Add_grad/SumSum$gradients/decoder/Relu_grad/ReluGrad0gradients/decoder/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

"gradients/decoder/Add_grad/ReshapeReshapegradients/decoder/Add_grad/Sum gradients/decoder/Add_grad/Shape*
T0*
Tshape0
§
 gradients/decoder/Add_grad/Sum_1Sum$gradients/decoder/Relu_grad/ReluGrad2gradients/decoder/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

$gradients/decoder/Add_grad/Reshape_1Reshape gradients/decoder/Add_grad/Sum_1"gradients/decoder/Add_grad/Shape_1*
T0*
Tshape0

$gradients/decoder/MatMul_grad/MatMulMatMul"gradients/decoder/Add_grad/Reshapedecoder/Variable/read*
transpose_b(*
T0*
transpose_a( 

&gradients/decoder/MatMul_grad/MatMul_1MatMulconcat_1"gradients/decoder/Add_grad/Reshape*
transpose_a(*
transpose_b( *
T0
F
gradients/concat_1_grad/RankConst*
dtype0*
value	B :
]
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
T0
R
gradients/concat_1_grad/ShapeConst*
valueB"d      *
dtype0
J
gradients/concat_1_grad/Shape_1Shapey_input*
T0*
out_type0
X
gradients/concat_1_grad/ShapeNShapeNAddy_input*
T0*
out_type0*
N

$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/ShapeN gradients/concat_1_grad/ShapeN:1*
N
¨
gradients/concat_1_grad/SliceSlice$gradients/decoder/MatMul_grad/MatMul$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/ShapeN*
T0*
Index0
®
gradients/concat_1_grad/Slice_1Slice$gradients/decoder/MatMul_grad/MatMul&gradients/concat_1_grad/ConcatOffset:1 gradients/concat_1_grad/ShapeN:1*
T0*
Index0
 
gradients/AddN_2AddN gradients/loss/Square_grad/Mul_1gradients/concat_1_grad/Slice*
T0*3
_class)
'%loc:@gradients/loss/Square_grad/Mul_1*
N
W
"gradients/encoder/Add_2_grad/ShapeConst*
valueB"d      *
dtype0
R
$gradients/encoder/Add_2_grad/Shape_1Const*
valueB:*
dtype0

2gradients/encoder/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/encoder/Add_2_grad/Shape$gradients/encoder/Add_2_grad/Shape_1*
T0

 gradients/encoder/Add_2_grad/SumSumgradients/AddN_22gradients/encoder/Add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

$gradients/encoder/Add_2_grad/ReshapeReshape gradients/encoder/Add_2_grad/Sum"gradients/encoder/Add_2_grad/Shape*
T0*
Tshape0

"gradients/encoder/Add_2_grad/Sum_1Sumgradients/AddN_24gradients/encoder/Add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

&gradients/encoder/Add_2_grad/Reshape_1Reshape"gradients/encoder/Add_2_grad/Sum_1$gradients/encoder/Add_2_grad/Shape_1*
T0*
Tshape0
V
gradients/Mul_1_grad/MulMulgradients/concat_1_grad/Slicerandom_normal*
T0
O
gradients/Mul_1_grad/Mul_1Mulgradients/concat_1_grad/SliceSqrt*
T0

$gradients/encoder/MatMul_grad/MatMulMatMul$gradients/encoder/Add_2_grad/Reshapeencoder/Variable_4/read*
T0*
transpose_a( *
transpose_b(
¦
&gradients/encoder/MatMul_grad/MatMul_1MatMulencoder/Flatten/flatten/Reshape$gradients/encoder/Add_2_grad/Reshape*
transpose_b( *
T0*
transpose_a(
Q
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mul_1_grad/Mul*
T0
I
gradients/Exp_grad/mulMulgradients/Sqrt_grad/SqrtGradExp*
T0
»
gradients/AddN_3AddN1gradients/loss/clip_by_value/Minimum_grad/Reshapegradients/Exp_grad/mul*
T0*D
_class:
86loc:@gradients/loss/clip_by_value/Minimum_grad/Reshape*
N
W
"gradients/encoder/Add_3_grad/ShapeConst*
valueB"d      *
dtype0
R
$gradients/encoder/Add_3_grad/Shape_1Const*
dtype0*
valueB:

2gradients/encoder/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/encoder/Add_3_grad/Shape$gradients/encoder/Add_3_grad/Shape_1*
T0

 gradients/encoder/Add_3_grad/SumSumgradients/AddN_32gradients/encoder/Add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

$gradients/encoder/Add_3_grad/ReshapeReshape gradients/encoder/Add_3_grad/Sum"gradients/encoder/Add_3_grad/Shape*
T0*
Tshape0

"gradients/encoder/Add_3_grad/Sum_1Sumgradients/AddN_34gradients/encoder/Add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

&gradients/encoder/Add_3_grad/Reshape_1Reshape"gradients/encoder/Add_3_grad/Sum_1$gradients/encoder/Add_3_grad/Shape_1*
T0*
Tshape0

&gradients/encoder/MatMul_1_grad/MatMulMatMul$gradients/encoder/Add_3_grad/Reshapeencoder/Variable_6/read*
T0*
transpose_a( *
transpose_b(
¨
(gradients/encoder/MatMul_1_grad/MatMul_1MatMulencoder/Flatten/flatten/Reshape$gradients/encoder/Add_3_grad/Reshape*
T0*
transpose_a(*
transpose_b( 
±
gradients/AddN_4AddN$gradients/encoder/MatMul_grad/MatMul&gradients/encoder/MatMul_1_grad/MatMul*
T0*7
_class-
+)loc:@gradients/encoder/MatMul_grad/MatMul*
N
q
4gradients/encoder/Flatten/flatten/Reshape_grad/ShapeConst*
dtype0*%
valueB"d             
 
6gradients/encoder/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN_44gradients/encoder/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0

&gradients/encoder/Relu_1_grad/ReluGradReluGrad6gradients/encoder/Flatten/flatten/Reshape_grad/Reshapeencoder/Relu_1*
T0
_
"gradients/encoder/Add_1_grad/ShapeConst*%
valueB"d             *
dtype0
R
$gradients/encoder/Add_1_grad/Shape_1Const*
dtype0*
valueB: 

2gradients/encoder/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/encoder/Add_1_grad/Shape$gradients/encoder/Add_1_grad/Shape_1*
T0
©
 gradients/encoder/Add_1_grad/SumSum&gradients/encoder/Relu_1_grad/ReluGrad2gradients/encoder/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

$gradients/encoder/Add_1_grad/ReshapeReshape gradients/encoder/Add_1_grad/Sum"gradients/encoder/Add_1_grad/Shape*
T0*
Tshape0
­
"gradients/encoder/Add_1_grad/Sum_1Sum&gradients/encoder/Relu_1_grad/ReluGrad4gradients/encoder/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

&gradients/encoder/Add_1_grad/Reshape_1Reshape"gradients/encoder/Add_1_grad/Sum_1$gradients/encoder/Add_1_grad/Shape_1*
T0*
Tshape0
y
&gradients/encoder/Conv2D_1_grad/ShapeNShapeNencoder/Reluencoder/Variable_2/read*
T0*
out_type0*
N
©
3gradients/encoder/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/encoder/Conv2D_1_grad/ShapeNencoder/Variable_2/read$gradients/encoder/Add_1_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

¢
4gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/Relu(gradients/encoder/Conv2D_1_grad/ShapeN:1$gradients/encoder/Add_1_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
|
$gradients/encoder/Relu_grad/ReluGradReluGrad3gradients/encoder/Conv2D_1_grad/Conv2DBackpropInputencoder/Relu*
T0
]
 gradients/encoder/Add_grad/ShapeConst*
dtype0*%
valueB"d            
P
"gradients/encoder/Add_grad/Shape_1Const*
valueB:*
dtype0

0gradients/encoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/encoder/Add_grad/Shape"gradients/encoder/Add_grad/Shape_1*
T0
£
gradients/encoder/Add_grad/SumSum$gradients/encoder/Relu_grad/ReluGrad0gradients/encoder/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 

"gradients/encoder/Add_grad/ReshapeReshapegradients/encoder/Add_grad/Sum gradients/encoder/Add_grad/Shape*
T0*
Tshape0
§
 gradients/encoder/Add_grad/Sum_1Sum$gradients/encoder/Relu_grad/ReluGrad2gradients/encoder/Add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 

$gradients/encoder/Add_grad/Reshape_1Reshape gradients/encoder/Add_grad/Sum_1"gradients/encoder/Add_grad/Shape_1*
T0*
Tshape0
o
$gradients/encoder/Conv2D_grad/ShapeNShapeNconcatencoder/Variable/read*
T0*
out_type0*
N
¡
1gradients/encoder/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/encoder/Conv2D_grad/ShapeNencoder/Variable/read"gradients/encoder/Add_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

2gradients/encoder/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconcat&gradients/encoder/Conv2D_grad/ShapeN:1"gradients/encoder/Add_grad/Reshape*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
 
global_norm/L2LossL2Loss2gradients/encoder/Conv2D_grad/Conv2DBackpropFilter*
T0*E
_class;
97loc:@gradients/encoder/Conv2D_grad/Conv2DBackpropFilter

global_norm/L2Loss_1L2Loss$gradients/encoder/Add_grad/Reshape_1*
T0*7
_class-
+)loc:@gradients/encoder/Add_grad/Reshape_1
¦
global_norm/L2Loss_2L2Loss4gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilter*
T0*G
_class=
;9loc:@gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilter

global_norm/L2Loss_3L2Loss&gradients/encoder/Add_1_grad/Reshape_1*
T0*9
_class/
-+loc:@gradients/encoder/Add_1_grad/Reshape_1

global_norm/L2Loss_4L2Loss&gradients/encoder/MatMul_grad/MatMul_1*
T0*9
_class/
-+loc:@gradients/encoder/MatMul_grad/MatMul_1

global_norm/L2Loss_5L2Loss&gradients/encoder/Add_2_grad/Reshape_1*
T0*9
_class/
-+loc:@gradients/encoder/Add_2_grad/Reshape_1

global_norm/L2Loss_6L2Loss(gradients/encoder/MatMul_1_grad/MatMul_1*
T0*;
_class1
/-loc:@gradients/encoder/MatMul_1_grad/MatMul_1

global_norm/L2Loss_7L2Loss&gradients/encoder/Add_3_grad/Reshape_1*
T0*9
_class/
-+loc:@gradients/encoder/Add_3_grad/Reshape_1

global_norm/L2Loss_8L2Loss&gradients/decoder/MatMul_grad/MatMul_1*
T0*9
_class/
-+loc:@gradients/decoder/MatMul_grad/MatMul_1

global_norm/L2Loss_9L2Loss$gradients/decoder/Add_grad/Reshape_1*
T0*7
_class-
+)loc:@gradients/decoder/Add_grad/Reshape_1
·
global_norm/L2Loss_10L2Loss<gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilter*
T0*O
_classE
CAloc:@gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilter

global_norm/L2Loss_11L2Loss&gradients/decoder/Add_1_grad/Reshape_1*
T0*9
_class/
-+loc:@gradients/decoder/Add_1_grad/Reshape_1
»
global_norm/L2Loss_12L2Loss>gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilter*
T0*Q
_classG
ECloc:@gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilter

global_norm/L2Loss_13L2Loss&gradients/decoder/Add_2_grad/Reshape_1*
T0*9
_class/
-+loc:@gradients/decoder/Add_2_grad/Reshape_1
í
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10global_norm/L2Loss_11global_norm/L2Loss_12global_norm/L2Loss_13*
T0*

axis *
N
?
global_norm/ConstConst*
valueB: *
dtype0
b
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
	keep_dims( *
T0
@
global_norm/Const_1Const*
dtype0*
valueB
 *   @
E
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0
9
global_norm/global_normSqrtglobal_norm/mul*
T0
¥
VerifyFinite/CheckNumericsCheckNumericsglobal_norm/global_norm**
messageFound Inf or NaN global norm.*
T0**
_class 
loc:@global_norm/global_norm

VerifyFinite/control_dependencyIdentityglobal_norm/global_norm^VerifyFinite/CheckNumerics*
T0**
_class 
loc:@global_norm/global_norm
J
clip_by_global_norm/truediv/xConst*
dtype0*
valueB
 *  ?
o
clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xVerifyFinite/control_dependency*
T0
F
clip_by_global_norm/ConstConst*
valueB
 *  ?*
dtype0
L
clip_by_global_norm/truediv_1/yConst*
valueB
 *  ?*
dtype0
m
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0
k
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0
F
clip_by_global_norm/mul/xConst*
dtype0*
valueB
 *  ?
_
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0
½
clip_by_global_norm/mul_1Mul2gradients/encoder/Conv2D_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*E
_class;
97loc:@gradients/encoder/Conv2D_grad/Conv2DBackpropFilter
¡
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*E
_class;
97loc:@gradients/encoder/Conv2D_grad/Conv2DBackpropFilter
¡
clip_by_global_norm/mul_2Mul$gradients/encoder/Add_grad/Reshape_1clip_by_global_norm/mul*
T0*7
_class-
+)loc:@gradients/encoder/Add_grad/Reshape_1

*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*7
_class-
+)loc:@gradients/encoder/Add_grad/Reshape_1
Á
clip_by_global_norm/mul_3Mul4gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*G
_class=
;9loc:@gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilter
£
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*G
_class=
;9loc:@gradients/encoder/Conv2D_1_grad/Conv2DBackpropFilter
¥
clip_by_global_norm/mul_4Mul&gradients/encoder/Add_1_grad/Reshape_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/encoder/Add_1_grad/Reshape_1

*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*9
_class/
-+loc:@gradients/encoder/Add_1_grad/Reshape_1
¥
clip_by_global_norm/mul_5Mul&gradients/encoder/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/encoder/MatMul_grad/MatMul_1

*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*9
_class/
-+loc:@gradients/encoder/MatMul_grad/MatMul_1
¥
clip_by_global_norm/mul_6Mul&gradients/encoder/Add_2_grad/Reshape_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/encoder/Add_2_grad/Reshape_1

*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*9
_class/
-+loc:@gradients/encoder/Add_2_grad/Reshape_1
©
clip_by_global_norm/mul_7Mul(gradients/encoder/MatMul_1_grad/MatMul_1clip_by_global_norm/mul*
T0*;
_class1
/-loc:@gradients/encoder/MatMul_1_grad/MatMul_1

*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*;
_class1
/-loc:@gradients/encoder/MatMul_1_grad/MatMul_1
¥
clip_by_global_norm/mul_8Mul&gradients/encoder/Add_3_grad/Reshape_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/encoder/Add_3_grad/Reshape_1

*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*9
_class/
-+loc:@gradients/encoder/Add_3_grad/Reshape_1
¥
clip_by_global_norm/mul_9Mul&gradients/decoder/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/decoder/MatMul_grad/MatMul_1

*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*9
_class/
-+loc:@gradients/decoder/MatMul_grad/MatMul_1
¢
clip_by_global_norm/mul_10Mul$gradients/decoder/Add_grad/Reshape_1clip_by_global_norm/mul*
T0*7
_class-
+)loc:@gradients/decoder/Add_grad/Reshape_1

*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*7
_class-
+)loc:@gradients/decoder/Add_grad/Reshape_1
Ò
clip_by_global_norm/mul_11Mul<gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*O
_classE
CAloc:@gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilter
­
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*
T0*O
_classE
CAloc:@gradients/decoder/conv2d_transpose_grad/Conv2DBackpropFilter
¦
clip_by_global_norm/mul_12Mul&gradients/decoder/Add_1_grad/Reshape_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/decoder/Add_1_grad/Reshape_1

+clip_by_global_norm/clip_by_global_norm/_11Identityclip_by_global_norm/mul_12*
T0*9
_class/
-+loc:@gradients/decoder/Add_1_grad/Reshape_1
Ö
clip_by_global_norm/mul_13Mul>gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*Q
_classG
ECloc:@gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilter
¯
+clip_by_global_norm/clip_by_global_norm/_12Identityclip_by_global_norm/mul_13*
T0*Q
_classG
ECloc:@gradients/decoder/conv2d_transpose_1_grad/Conv2DBackpropFilter
¦
clip_by_global_norm/mul_14Mul&gradients/decoder/Add_2_grad/Reshape_1clip_by_global_norm/mul*
T0*9
_class/
-+loc:@gradients/decoder/Add_2_grad/Reshape_1

+clip_by_global_norm/clip_by_global_norm/_13Identityclip_by_global_norm/mul_14*
T0*9
_class/
-+loc:@gradients/decoder/Add_2_grad/Reshape_1
k
beta1_power/initial_valueConst*
valueB
 *fff?*#
_class
loc:@decoder/Variable*
dtype0
|
beta1_power
VariableV2*#
_class
loc:@decoder/Variable*
dtype0*
	container *
shape: *
shared_name 

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*#
_class
loc:@decoder/Variable*
validate_shape(
W
beta1_power/readIdentitybeta1_power*
T0*#
_class
loc:@decoder/Variable
k
beta2_power/initial_valueConst*
dtype0*
valueB
 *w¾?*#
_class
loc:@decoder/Variable
|
beta2_power
VariableV2*
shape: *
shared_name *#
_class
loc:@decoder/Variable*
dtype0*
	container 

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*#
_class
loc:@decoder/Variable*
validate_shape(*
use_locking(
W
beta2_power/readIdentitybeta2_power*
T0*#
_class
loc:@decoder/Variable

7encoder/Variable/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"            *#
_class
loc:@encoder/Variable*
dtype0

-encoder/Variable/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@encoder/Variable*
dtype0
×
'encoder/Variable/Adam/Initializer/zerosFill7encoder/Variable/Adam/Initializer/zeros/shape_as_tensor-encoder/Variable/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@encoder/Variable

encoder/Variable/Adam
VariableV2*#
_class
loc:@encoder/Variable*
dtype0*
	container *
shape:*
shared_name 
½
encoder/Variable/Adam/AssignAssignencoder/Variable/Adam'encoder/Variable/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@encoder/Variable*
validate_shape(
k
encoder/Variable/Adam/readIdentityencoder/Variable/Adam*
T0*#
_class
loc:@encoder/Variable

9encoder/Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *#
_class
loc:@encoder/Variable*
dtype0

/encoder/Variable/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@encoder/Variable*
dtype0
Ý
)encoder/Variable/Adam_1/Initializer/zerosFill9encoder/Variable/Adam_1/Initializer/zeros/shape_as_tensor/encoder/Variable/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@encoder/Variable

encoder/Variable/Adam_1
VariableV2*#
_class
loc:@encoder/Variable*
dtype0*
	container *
shape:*
shared_name 
Ã
encoder/Variable/Adam_1/AssignAssignencoder/Variable/Adam_1)encoder/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@encoder/Variable*
validate_shape(
o
encoder/Variable/Adam_1/readIdentityencoder/Variable/Adam_1*
T0*#
_class
loc:@encoder/Variable

)encoder/Variable_1/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/Variable_1*
dtype0

encoder/Variable_1/Adam
VariableV2*
shape:*
shared_name *%
_class
loc:@encoder/Variable_1*
dtype0*
	container 
Å
encoder/Variable_1/Adam/AssignAssignencoder/Variable_1/Adam)encoder/Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1*
validate_shape(
q
encoder/Variable_1/Adam/readIdentityencoder/Variable_1/Adam*
T0*%
_class
loc:@encoder/Variable_1

+encoder/Variable_1/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/Variable_1*
dtype0

encoder/Variable_1/Adam_1
VariableV2*%
_class
loc:@encoder/Variable_1*
dtype0*
	container *
shape:*
shared_name 
Ë
 encoder/Variable_1/Adam_1/AssignAssignencoder/Variable_1/Adam_1+encoder/Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1
u
encoder/Variable_1/Adam_1/readIdentityencoder/Variable_1/Adam_1*
T0*%
_class
loc:@encoder/Variable_1

9encoder/Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"             *%
_class
loc:@encoder/Variable_2*
dtype0

/encoder/Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@encoder/Variable_2*
dtype0
ß
)encoder/Variable_2/Adam/Initializer/zerosFill9encoder/Variable_2/Adam/Initializer/zeros/shape_as_tensor/encoder/Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_2

encoder/Variable_2/Adam
VariableV2*%
_class
loc:@encoder/Variable_2*
dtype0*
	container *
shape: *
shared_name 
Å
encoder/Variable_2/Adam/AssignAssignencoder/Variable_2/Adam)encoder/Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_2
q
encoder/Variable_2/Adam/readIdentityencoder/Variable_2/Adam*
T0*%
_class
loc:@encoder/Variable_2

;encoder/Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             *%
_class
loc:@encoder/Variable_2*
dtype0

1encoder/Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@encoder/Variable_2*
dtype0
å
+encoder/Variable_2/Adam_1/Initializer/zerosFill;encoder/Variable_2/Adam_1/Initializer/zeros/shape_as_tensor1encoder/Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_2

encoder/Variable_2/Adam_1
VariableV2*%
_class
loc:@encoder/Variable_2*
dtype0*
	container *
shape: *
shared_name 
Ë
 encoder/Variable_2/Adam_1/AssignAssignencoder/Variable_2/Adam_1+encoder/Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_2
u
encoder/Variable_2/Adam_1/readIdentityencoder/Variable_2/Adam_1*
T0*%
_class
loc:@encoder/Variable_2

)encoder/Variable_3/Adam/Initializer/zerosConst*
valueB *    *%
_class
loc:@encoder/Variable_3*
dtype0

encoder/Variable_3/Adam
VariableV2*
shape: *
shared_name *%
_class
loc:@encoder/Variable_3*
dtype0*
	container 
Å
encoder/Variable_3/Adam/AssignAssignencoder/Variable_3/Adam)encoder/Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(
q
encoder/Variable_3/Adam/readIdentityencoder/Variable_3/Adam*
T0*%
_class
loc:@encoder/Variable_3

+encoder/Variable_3/Adam_1/Initializer/zerosConst*
valueB *    *%
_class
loc:@encoder/Variable_3*
dtype0

encoder/Variable_3/Adam_1
VariableV2*
shape: *
shared_name *%
_class
loc:@encoder/Variable_3*
dtype0*
	container 
Ë
 encoder/Variable_3/Adam_1/AssignAssignencoder/Variable_3/Adam_1+encoder/Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(
u
encoder/Variable_3/Adam_1/readIdentityencoder/Variable_3/Adam_1*
T0*%
_class
loc:@encoder/Variable_3

9encoder/Variable_4/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *%
_class
loc:@encoder/Variable_4*
dtype0

/encoder/Variable_4/Adam/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@encoder/Variable_4*
dtype0
ß
)encoder/Variable_4/Adam/Initializer/zerosFill9encoder/Variable_4/Adam/Initializer/zeros/shape_as_tensor/encoder/Variable_4/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_4

encoder/Variable_4/Adam
VariableV2*
dtype0*
	container *
shape:	*
shared_name *%
_class
loc:@encoder/Variable_4
Å
encoder/Variable_4/Adam/AssignAssignencoder/Variable_4/Adam)encoder/Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4*
validate_shape(
q
encoder/Variable_4/Adam/readIdentityencoder/Variable_4/Adam*
T0*%
_class
loc:@encoder/Variable_4

;encoder/Variable_4/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *%
_class
loc:@encoder/Variable_4*
dtype0

1encoder/Variable_4/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@encoder/Variable_4*
dtype0
å
+encoder/Variable_4/Adam_1/Initializer/zerosFill;encoder/Variable_4/Adam_1/Initializer/zeros/shape_as_tensor1encoder/Variable_4/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_4

encoder/Variable_4/Adam_1
VariableV2*
dtype0*
	container *
shape:	*
shared_name *%
_class
loc:@encoder/Variable_4
Ë
 encoder/Variable_4/Adam_1/AssignAssignencoder/Variable_4/Adam_1+encoder/Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4
u
encoder/Variable_4/Adam_1/readIdentityencoder/Variable_4/Adam_1*
T0*%
_class
loc:@encoder/Variable_4

)encoder/Variable_5/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/Variable_5*
dtype0

encoder/Variable_5/Adam
VariableV2*
shape:*
shared_name *%
_class
loc:@encoder/Variable_5*
dtype0*
	container 
Å
encoder/Variable_5/Adam/AssignAssignencoder/Variable_5/Adam)encoder/Variable_5/Adam/Initializer/zeros*
T0*%
_class
loc:@encoder/Variable_5*
validate_shape(*
use_locking(
q
encoder/Variable_5/Adam/readIdentityencoder/Variable_5/Adam*
T0*%
_class
loc:@encoder/Variable_5

+encoder/Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@encoder/Variable_5

encoder/Variable_5/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *%
_class
loc:@encoder/Variable_5
Ë
 encoder/Variable_5/Adam_1/AssignAssignencoder/Variable_5/Adam_1+encoder/Variable_5/Adam_1/Initializer/zeros*
T0*%
_class
loc:@encoder/Variable_5*
validate_shape(*
use_locking(
u
encoder/Variable_5/Adam_1/readIdentityencoder/Variable_5/Adam_1*
T0*%
_class
loc:@encoder/Variable_5

9encoder/Variable_6/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"      *%
_class
loc:@encoder/Variable_6

/encoder/Variable_6/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *%
_class
loc:@encoder/Variable_6
ß
)encoder/Variable_6/Adam/Initializer/zerosFill9encoder/Variable_6/Adam/Initializer/zeros/shape_as_tensor/encoder/Variable_6/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_6

encoder/Variable_6/Adam
VariableV2*%
_class
loc:@encoder/Variable_6*
dtype0*
	container *
shape:	*
shared_name 
Å
encoder/Variable_6/Adam/AssignAssignencoder/Variable_6/Adam)encoder/Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(
q
encoder/Variable_6/Adam/readIdentityencoder/Variable_6/Adam*
T0*%
_class
loc:@encoder/Variable_6

;encoder/Variable_6/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *%
_class
loc:@encoder/Variable_6*
dtype0

1encoder/Variable_6/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@encoder/Variable_6*
dtype0
å
+encoder/Variable_6/Adam_1/Initializer/zerosFill;encoder/Variable_6/Adam_1/Initializer/zeros/shape_as_tensor1encoder/Variable_6/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@encoder/Variable_6

encoder/Variable_6/Adam_1
VariableV2*%
_class
loc:@encoder/Variable_6*
dtype0*
	container *
shape:	*
shared_name 
Ë
 encoder/Variable_6/Adam_1/AssignAssignencoder/Variable_6/Adam_1+encoder/Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(
u
encoder/Variable_6/Adam_1/readIdentityencoder/Variable_6/Adam_1*
T0*%
_class
loc:@encoder/Variable_6

)encoder/Variable_7/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/Variable_7*
dtype0

encoder/Variable_7/Adam
VariableV2*%
_class
loc:@encoder/Variable_7*
dtype0*
	container *
shape:*
shared_name 
Å
encoder/Variable_7/Adam/AssignAssignencoder/Variable_7/Adam)encoder/Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(
q
encoder/Variable_7/Adam/readIdentityencoder/Variable_7/Adam*
T0*%
_class
loc:@encoder/Variable_7

+encoder/Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/Variable_7*
dtype0

encoder/Variable_7/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *%
_class
loc:@encoder/Variable_7
Ë
 encoder/Variable_7/Adam_1/AssignAssignencoder/Variable_7/Adam_1+encoder/Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(
u
encoder/Variable_7/Adam_1/readIdentityencoder/Variable_7/Adam_1*
T0*%
_class
loc:@encoder/Variable_7

7decoder/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"     *#
_class
loc:@decoder/Variable*
dtype0

-decoder/Variable/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *#
_class
loc:@decoder/Variable
×
'decoder/Variable/Adam/Initializer/zerosFill7decoder/Variable/Adam/Initializer/zeros/shape_as_tensor-decoder/Variable/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@decoder/Variable

decoder/Variable/Adam
VariableV2*#
_class
loc:@decoder/Variable*
dtype0*
	container *
shape:	*
shared_name 
½
decoder/Variable/Adam/AssignAssigndecoder/Variable/Adam'decoder/Variable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*#
_class
loc:@decoder/Variable
k
decoder/Variable/Adam/readIdentitydecoder/Variable/Adam*
T0*#
_class
loc:@decoder/Variable

9decoder/Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"     *#
_class
loc:@decoder/Variable*
dtype0

/decoder/Variable/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *#
_class
loc:@decoder/Variable
Ý
)decoder/Variable/Adam_1/Initializer/zerosFill9decoder/Variable/Adam_1/Initializer/zeros/shape_as_tensor/decoder/Variable/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@decoder/Variable

decoder/Variable/Adam_1
VariableV2*
shared_name *#
_class
loc:@decoder/Variable*
dtype0*
	container *
shape:	
Ã
decoder/Variable/Adam_1/AssignAssigndecoder/Variable/Adam_1)decoder/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@decoder/Variable*
validate_shape(
o
decoder/Variable/Adam_1/readIdentitydecoder/Variable/Adam_1*
T0*#
_class
loc:@decoder/Variable

)decoder/Variable_1/Adam/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@decoder/Variable_1

decoder/Variable_1/Adam
VariableV2*%
_class
loc:@decoder/Variable_1*
dtype0*
	container *
shape:*
shared_name 
Å
decoder/Variable_1/Adam/AssignAssigndecoder/Variable_1/Adam)decoder/Variable_1/Adam/Initializer/zeros*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(*
use_locking(
q
decoder/Variable_1/Adam/readIdentitydecoder/Variable_1/Adam*
T0*%
_class
loc:@decoder/Variable_1

+decoder/Variable_1/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@decoder/Variable_1*
dtype0

decoder/Variable_1/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *%
_class
loc:@decoder/Variable_1
Ë
 decoder/Variable_1/Adam_1/AssignAssigndecoder/Variable_1/Adam_1+decoder/Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(
u
decoder/Variable_1/Adam_1/readIdentitydecoder/Variable_1/Adam_1*
T0*%
_class
loc:@decoder/Variable_1

9decoder/Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"            *%
_class
loc:@decoder/Variable_2*
dtype0

/decoder/Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@decoder/Variable_2*
dtype0
ß
)decoder/Variable_2/Adam/Initializer/zerosFill9decoder/Variable_2/Adam/Initializer/zeros/shape_as_tensor/decoder/Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@decoder/Variable_2

decoder/Variable_2/Adam
VariableV2*
shape:*
shared_name *%
_class
loc:@decoder/Variable_2*
dtype0*
	container 
Å
decoder/Variable_2/Adam/AssignAssigndecoder/Variable_2/Adam)decoder/Variable_2/Adam/Initializer/zeros*
T0*%
_class
loc:@decoder/Variable_2*
validate_shape(*
use_locking(
q
decoder/Variable_2/Adam/readIdentitydecoder/Variable_2/Adam*
T0*%
_class
loc:@decoder/Variable_2

;decoder/Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *%
_class
loc:@decoder/Variable_2*
dtype0

1decoder/Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@decoder/Variable_2*
dtype0
å
+decoder/Variable_2/Adam_1/Initializer/zerosFill;decoder/Variable_2/Adam_1/Initializer/zeros/shape_as_tensor1decoder/Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@decoder/Variable_2

decoder/Variable_2/Adam_1
VariableV2*
shared_name *%
_class
loc:@decoder/Variable_2*
dtype0*
	container *
shape:
Ë
 decoder/Variable_2/Adam_1/AssignAssigndecoder/Variable_2/Adam_1+decoder/Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@decoder/Variable_2
u
decoder/Variable_2/Adam_1/readIdentitydecoder/Variable_2/Adam_1*
T0*%
_class
loc:@decoder/Variable_2

)decoder/Variable_3/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@decoder/Variable_3*
dtype0

decoder/Variable_3/Adam
VariableV2*
shared_name *%
_class
loc:@decoder/Variable_3*
dtype0*
	container *
shape:
Å
decoder/Variable_3/Adam/AssignAssigndecoder/Variable_3/Adam)decoder/Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(
q
decoder/Variable_3/Adam/readIdentitydecoder/Variable_3/Adam*
T0*%
_class
loc:@decoder/Variable_3

+decoder/Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@decoder/Variable_3

decoder/Variable_3/Adam_1
VariableV2*
shared_name *%
_class
loc:@decoder/Variable_3*
dtype0*
	container *
shape:
Ë
 decoder/Variable_3/Adam_1/AssignAssigndecoder/Variable_3/Adam_1+decoder/Variable_3/Adam_1/Initializer/zeros*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(*
use_locking(
u
decoder/Variable_3/Adam_1/readIdentitydecoder/Variable_3/Adam_1*
T0*%
_class
loc:@decoder/Variable_3

)decoder/Variable_4/Adam/Initializer/zerosConst*%
valueB*    *%
_class
loc:@decoder/Variable_4*
dtype0

decoder/Variable_4/Adam
VariableV2*
shape:*
shared_name *%
_class
loc:@decoder/Variable_4*
dtype0*
	container 
Å
decoder/Variable_4/Adam/AssignAssigndecoder/Variable_4/Adam)decoder/Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4
q
decoder/Variable_4/Adam/readIdentitydecoder/Variable_4/Adam*
T0*%
_class
loc:@decoder/Variable_4

+decoder/Variable_4/Adam_1/Initializer/zerosConst*%
valueB*    *%
_class
loc:@decoder/Variable_4*
dtype0

decoder/Variable_4/Adam_1
VariableV2*%
_class
loc:@decoder/Variable_4*
dtype0*
	container *
shape:*
shared_name 
Ë
 decoder/Variable_4/Adam_1/AssignAssigndecoder/Variable_4/Adam_1+decoder/Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4*
validate_shape(
u
decoder/Variable_4/Adam_1/readIdentitydecoder/Variable_4/Adam_1*
T0*%
_class
loc:@decoder/Variable_4

)decoder/Variable_5/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@decoder/Variable_5*
dtype0

decoder/Variable_5/Adam
VariableV2*
shape:*
shared_name *%
_class
loc:@decoder/Variable_5*
dtype0*
	container 
Å
decoder/Variable_5/Adam/AssignAssigndecoder/Variable_5/Adam)decoder/Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/Variable_5*
validate_shape(
q
decoder/Variable_5/Adam/readIdentitydecoder/Variable_5/Adam*
T0*%
_class
loc:@decoder/Variable_5

+decoder/Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@decoder/Variable_5

decoder/Variable_5/Adam_1
VariableV2*
shared_name *%
_class
loc:@decoder/Variable_5*
dtype0*
	container *
shape:
Ë
 decoder/Variable_5/Adam_1/AssignAssigndecoder/Variable_5/Adam_1+decoder/Variable_5/Adam_1/Initializer/zeros*
T0*%
_class
loc:@decoder/Variable_5*
validate_shape(*
use_locking(
u
decoder/Variable_5/Adam_1/readIdentitydecoder/Variable_5/Adam_1*
T0*%
_class
loc:@decoder/Variable_5
?
Adam/learning_rateConst*
valueB
 *o:*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w¾?*
dtype0
9
Adam/epsilonConst*
dtype0*
valueB
 *wÌ+2
Ô
&Adam/update_encoder/Variable/ApplyAdam	ApplyAdamencoder/Variableencoder/Variable/Adamencoder/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
T0*#
_class
loc:@encoder/Variable*
use_nesterov( 
Þ
(Adam/update_encoder/Variable_1/ApplyAdam	ApplyAdamencoder/Variable_1encoder/Variable_1/Adamencoder/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*%
_class
loc:@encoder/Variable_1*
use_nesterov( 
Þ
(Adam/update_encoder/Variable_2/ApplyAdam	ApplyAdamencoder/Variable_2encoder/Variable_2/Adamencoder/Variable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0*%
_class
loc:@encoder/Variable_2*
use_nesterov( 
Þ
(Adam/update_encoder/Variable_3/ApplyAdam	ApplyAdamencoder/Variable_3encoder/Variable_3/Adamencoder/Variable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_nesterov( *
use_locking( *
T0*%
_class
loc:@encoder/Variable_3
Þ
(Adam/update_encoder/Variable_4/ApplyAdam	ApplyAdamencoder/Variable_4encoder/Variable_4/Adamencoder/Variable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
T0*%
_class
loc:@encoder/Variable_4*
use_nesterov( 
Þ
(Adam/update_encoder/Variable_5/ApplyAdam	ApplyAdamencoder/Variable_5encoder/Variable_5/Adamencoder/Variable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_nesterov( *
use_locking( *
T0*%
_class
loc:@encoder/Variable_5
Þ
(Adam/update_encoder/Variable_6/ApplyAdam	ApplyAdamencoder/Variable_6encoder/Variable_6/Adamencoder/Variable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_nesterov( *
use_locking( *
T0*%
_class
loc:@encoder/Variable_6
Þ
(Adam/update_encoder/Variable_7/ApplyAdam	ApplyAdamencoder/Variable_7encoder/Variable_7/Adamencoder/Variable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*%
_class
loc:@encoder/Variable_7*
use_nesterov( 
Ô
&Adam/update_decoder/Variable/ApplyAdam	ApplyAdamdecoder/Variabledecoder/Variable/Adamdecoder/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
use_locking( *
T0*#
_class
loc:@decoder/Variable*
use_nesterov( 
Þ
(Adam/update_decoder/Variable_1/ApplyAdam	ApplyAdamdecoder/Variable_1decoder/Variable_1/Adamdecoder/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
use_locking( *
T0*%
_class
loc:@decoder/Variable_1*
use_nesterov( 
ß
(Adam/update_decoder/Variable_2/ApplyAdam	ApplyAdamdecoder/Variable_2decoder/Variable_2/Adamdecoder/Variable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
T0*%
_class
loc:@decoder/Variable_2*
use_nesterov( *
use_locking( 
ß
(Adam/update_decoder/Variable_3/ApplyAdam	ApplyAdamdecoder/Variable_3decoder/Variable_3/Adamdecoder/Variable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_11*
use_nesterov( *
use_locking( *
T0*%
_class
loc:@decoder/Variable_3
ß
(Adam/update_decoder/Variable_4/ApplyAdam	ApplyAdamdecoder/Variable_4decoder/Variable_4/Adamdecoder/Variable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_12*
T0*%
_class
loc:@decoder/Variable_4*
use_nesterov( *
use_locking( 
ß
(Adam/update_decoder/Variable_5/ApplyAdam	ApplyAdamdecoder/Variable_5decoder/Variable_5/Adamdecoder/Variable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_13*
use_locking( *
T0*%
_class
loc:@decoder/Variable_5*
use_nesterov( 
±
Adam/mulMulbeta1_power/read
Adam/beta1'^Adam/update_decoder/Variable/ApplyAdam)^Adam/update_decoder/Variable_1/ApplyAdam)^Adam/update_decoder/Variable_2/ApplyAdam)^Adam/update_decoder/Variable_3/ApplyAdam)^Adam/update_decoder/Variable_4/ApplyAdam)^Adam/update_decoder/Variable_5/ApplyAdam'^Adam/update_encoder/Variable/ApplyAdam)^Adam/update_encoder/Variable_1/ApplyAdam)^Adam/update_encoder/Variable_2/ApplyAdam)^Adam/update_encoder/Variable_3/ApplyAdam)^Adam/update_encoder/Variable_4/ApplyAdam)^Adam/update_encoder/Variable_5/ApplyAdam)^Adam/update_encoder/Variable_6/ApplyAdam)^Adam/update_encoder/Variable_7/ApplyAdam*
T0*#
_class
loc:@decoder/Variable

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
T0*#
_class
loc:@decoder/Variable
³

Adam/mul_1Mulbeta2_power/read
Adam/beta2'^Adam/update_decoder/Variable/ApplyAdam)^Adam/update_decoder/Variable_1/ApplyAdam)^Adam/update_decoder/Variable_2/ApplyAdam)^Adam/update_decoder/Variable_3/ApplyAdam)^Adam/update_decoder/Variable_4/ApplyAdam)^Adam/update_decoder/Variable_5/ApplyAdam'^Adam/update_encoder/Variable/ApplyAdam)^Adam/update_encoder/Variable_1/ApplyAdam)^Adam/update_encoder/Variable_2/ApplyAdam)^Adam/update_encoder/Variable_3/ApplyAdam)^Adam/update_encoder/Variable_4/ApplyAdam)^Adam/update_encoder/Variable_5/ApplyAdam)^Adam/update_encoder/Variable_6/ApplyAdam)^Adam/update_encoder/Variable_7/ApplyAdam*
T0*#
_class
loc:@decoder/Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*#
_class
loc:@decoder/Variable*
validate_shape(

AdamNoOp^Adam/Assign^Adam/Assign_1'^Adam/update_decoder/Variable/ApplyAdam)^Adam/update_decoder/Variable_1/ApplyAdam)^Adam/update_decoder/Variable_2/ApplyAdam)^Adam/update_decoder/Variable_3/ApplyAdam)^Adam/update_decoder/Variable_4/ApplyAdam)^Adam/update_decoder/Variable_5/ApplyAdam'^Adam/update_encoder/Variable/ApplyAdam)^Adam/update_encoder/Variable_1/ApplyAdam)^Adam/update_encoder/Variable_2/ApplyAdam)^Adam/update_encoder/Variable_3/ApplyAdam)^Adam/update_encoder/Variable_4/ApplyAdam)^Adam/update_encoder/Variable_5/ApplyAdam)^Adam/update_encoder/Variable_6/ApplyAdam)^Adam/update_encoder/Variable_7/ApplyAdam
8

save/ConstConst*
valueB Bmodel*
dtype0
Ä
save/SaveV2/tensor_namesConst*
dtype0*
valueB,Bbeta1_powerBbeta2_powerBdecoder/VariableBdecoder/Variable/AdamBdecoder/Variable/Adam_1Bdecoder/Variable_1Bdecoder/Variable_1/AdamBdecoder/Variable_1/Adam_1Bdecoder/Variable_2Bdecoder/Variable_2/AdamBdecoder/Variable_2/Adam_1Bdecoder/Variable_3Bdecoder/Variable_3/AdamBdecoder/Variable_3/Adam_1Bdecoder/Variable_4Bdecoder/Variable_4/AdamBdecoder/Variable_4/Adam_1Bdecoder/Variable_5Bdecoder/Variable_5/AdamBdecoder/Variable_5/Adam_1Bencoder/VariableBencoder/Variable/AdamBencoder/Variable/Adam_1Bencoder/Variable_1Bencoder/Variable_1/AdamBencoder/Variable_1/Adam_1Bencoder/Variable_2Bencoder/Variable_2/AdamBencoder/Variable_2/Adam_1Bencoder/Variable_3Bencoder/Variable_3/AdamBencoder/Variable_3/Adam_1Bencoder/Variable_4Bencoder/Variable_4/AdamBencoder/Variable_4/Adam_1Bencoder/Variable_5Bencoder/Variable_5/AdamBencoder/Variable_5/Adam_1Bencoder/Variable_6Bencoder/Variable_6/AdamBencoder/Variable_6/Adam_1Bencoder/Variable_7Bencoder/Variable_7/AdamBencoder/Variable_7/Adam_1

save/SaveV2/shape_and_slicesConst*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
	
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerdecoder/Variabledecoder/Variable/Adamdecoder/Variable/Adam_1decoder/Variable_1decoder/Variable_1/Adamdecoder/Variable_1/Adam_1decoder/Variable_2decoder/Variable_2/Adamdecoder/Variable_2/Adam_1decoder/Variable_3decoder/Variable_3/Adamdecoder/Variable_3/Adam_1decoder/Variable_4decoder/Variable_4/Adamdecoder/Variable_4/Adam_1decoder/Variable_5decoder/Variable_5/Adamdecoder/Variable_5/Adam_1encoder/Variableencoder/Variable/Adamencoder/Variable/Adam_1encoder/Variable_1encoder/Variable_1/Adamencoder/Variable_1/Adam_1encoder/Variable_2encoder/Variable_2/Adamencoder/Variable_2/Adam_1encoder/Variable_3encoder/Variable_3/Adamencoder/Variable_3/Adam_1encoder/Variable_4encoder/Variable_4/Adamencoder/Variable_4/Adam_1encoder/Variable_5encoder/Variable_5/Adamencoder/Variable_5/Adam_1encoder/Variable_6encoder/Variable_6/Adamencoder/Variable_6/Adam_1encoder/Variable_7encoder/Variable_7/Adamencoder/Variable_7/Adam_1*:
dtypes0
.2,
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
Ö
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueB,Bbeta1_powerBbeta2_powerBdecoder/VariableBdecoder/Variable/AdamBdecoder/Variable/Adam_1Bdecoder/Variable_1Bdecoder/Variable_1/AdamBdecoder/Variable_1/Adam_1Bdecoder/Variable_2Bdecoder/Variable_2/AdamBdecoder/Variable_2/Adam_1Bdecoder/Variable_3Bdecoder/Variable_3/AdamBdecoder/Variable_3/Adam_1Bdecoder/Variable_4Bdecoder/Variable_4/AdamBdecoder/Variable_4/Adam_1Bdecoder/Variable_5Bdecoder/Variable_5/AdamBdecoder/Variable_5/Adam_1Bencoder/VariableBencoder/Variable/AdamBencoder/Variable/Adam_1Bencoder/Variable_1Bencoder/Variable_1/AdamBencoder/Variable_1/Adam_1Bencoder/Variable_2Bencoder/Variable_2/AdamBencoder/Variable_2/Adam_1Bencoder/Variable_3Bencoder/Variable_3/AdamBencoder/Variable_3/Adam_1Bencoder/Variable_4Bencoder/Variable_4/AdamBencoder/Variable_4/Adam_1Bencoder/Variable_5Bencoder/Variable_5/AdamBencoder/Variable_5/Adam_1Bencoder/Variable_6Bencoder/Variable_6/AdamBencoder/Variable_6/Adam_1Bencoder/Variable_7Bencoder/Variable_7/AdamBencoder/Variable_7/Adam_1*
dtype0
±
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
°
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*:
dtypes0
.2,

save/AssignAssignbeta1_powersave/RestoreV2*
T0*#
_class
loc:@decoder/Variable*
validate_shape(*
use_locking(

save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*#
_class
loc:@decoder/Variable*
validate_shape(*
use_locking(

save/Assign_2Assigndecoder/Variablesave/RestoreV2:2*
use_locking(*
T0*#
_class
loc:@decoder/Variable*
validate_shape(

save/Assign_3Assigndecoder/Variable/Adamsave/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@decoder/Variable*
validate_shape(

save/Assign_4Assigndecoder/Variable/Adam_1save/RestoreV2:4*
use_locking(*
T0*#
_class
loc:@decoder/Variable*
validate_shape(

save/Assign_5Assigndecoder/Variable_1save/RestoreV2:5*
use_locking(*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(

save/Assign_6Assigndecoder/Variable_1/Adamsave/RestoreV2:6*
use_locking(*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(

save/Assign_7Assigndecoder/Variable_1/Adam_1save/RestoreV2:7*
use_locking(*
T0*%
_class
loc:@decoder/Variable_1*
validate_shape(

save/Assign_8Assigndecoder/Variable_2save/RestoreV2:8*
use_locking(*
T0*%
_class
loc:@decoder/Variable_2*
validate_shape(

save/Assign_9Assigndecoder/Variable_2/Adamsave/RestoreV2:9*
T0*%
_class
loc:@decoder/Variable_2*
validate_shape(*
use_locking(

save/Assign_10Assigndecoder/Variable_2/Adam_1save/RestoreV2:10*
use_locking(*
T0*%
_class
loc:@decoder/Variable_2*
validate_shape(

save/Assign_11Assigndecoder/Variable_3save/RestoreV2:11*
use_locking(*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(

save/Assign_12Assigndecoder/Variable_3/Adamsave/RestoreV2:12*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(*
use_locking(

save/Assign_13Assigndecoder/Variable_3/Adam_1save/RestoreV2:13*
T0*%
_class
loc:@decoder/Variable_3*
validate_shape(*
use_locking(

save/Assign_14Assigndecoder/Variable_4save/RestoreV2:14*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4*
validate_shape(

save/Assign_15Assigndecoder/Variable_4/Adamsave/RestoreV2:15*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4*
validate_shape(

save/Assign_16Assigndecoder/Variable_4/Adam_1save/RestoreV2:16*
validate_shape(*
use_locking(*
T0*%
_class
loc:@decoder/Variable_4

save/Assign_17Assigndecoder/Variable_5save/RestoreV2:17*
use_locking(*
T0*%
_class
loc:@decoder/Variable_5*
validate_shape(

save/Assign_18Assigndecoder/Variable_5/Adamsave/RestoreV2:18*
use_locking(*
T0*%
_class
loc:@decoder/Variable_5*
validate_shape(

save/Assign_19Assigndecoder/Variable_5/Adam_1save/RestoreV2:19*
validate_shape(*
use_locking(*
T0*%
_class
loc:@decoder/Variable_5

save/Assign_20Assignencoder/Variablesave/RestoreV2:20*
validate_shape(*
use_locking(*
T0*#
_class
loc:@encoder/Variable

save/Assign_21Assignencoder/Variable/Adamsave/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@encoder/Variable*
validate_shape(

save/Assign_22Assignencoder/Variable/Adam_1save/RestoreV2:22*
T0*#
_class
loc:@encoder/Variable*
validate_shape(*
use_locking(

save/Assign_23Assignencoder/Variable_1save/RestoreV2:23*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1*
validate_shape(

save/Assign_24Assignencoder/Variable_1/Adamsave/RestoreV2:24*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1*
validate_shape(

save/Assign_25Assignencoder/Variable_1/Adam_1save/RestoreV2:25*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_1

save/Assign_26Assignencoder/Variable_2save/RestoreV2:26*
T0*%
_class
loc:@encoder/Variable_2*
validate_shape(*
use_locking(

save/Assign_27Assignencoder/Variable_2/Adamsave/RestoreV2:27*
use_locking(*
T0*%
_class
loc:@encoder/Variable_2*
validate_shape(

save/Assign_28Assignencoder/Variable_2/Adam_1save/RestoreV2:28*
use_locking(*
T0*%
_class
loc:@encoder/Variable_2*
validate_shape(

save/Assign_29Assignencoder/Variable_3save/RestoreV2:29*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(*
use_locking(

save/Assign_30Assignencoder/Variable_3/Adamsave/RestoreV2:30*
use_locking(*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(

save/Assign_31Assignencoder/Variable_3/Adam_1save/RestoreV2:31*
use_locking(*
T0*%
_class
loc:@encoder/Variable_3*
validate_shape(

save/Assign_32Assignencoder/Variable_4save/RestoreV2:32*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4*
validate_shape(

save/Assign_33Assignencoder/Variable_4/Adamsave/RestoreV2:33*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4*
validate_shape(

save/Assign_34Assignencoder/Variable_4/Adam_1save/RestoreV2:34*
use_locking(*
T0*%
_class
loc:@encoder/Variable_4*
validate_shape(

save/Assign_35Assignencoder/Variable_5save/RestoreV2:35*
validate_shape(*
use_locking(*
T0*%
_class
loc:@encoder/Variable_5

save/Assign_36Assignencoder/Variable_5/Adamsave/RestoreV2:36*
T0*%
_class
loc:@encoder/Variable_5*
validate_shape(*
use_locking(

save/Assign_37Assignencoder/Variable_5/Adam_1save/RestoreV2:37*
use_locking(*
T0*%
_class
loc:@encoder/Variable_5*
validate_shape(

save/Assign_38Assignencoder/Variable_6save/RestoreV2:38*
use_locking(*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(

save/Assign_39Assignencoder/Variable_6/Adamsave/RestoreV2:39*
use_locking(*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(

save/Assign_40Assignencoder/Variable_6/Adam_1save/RestoreV2:40*
T0*%
_class
loc:@encoder/Variable_6*
validate_shape(*
use_locking(

save/Assign_41Assignencoder/Variable_7save/RestoreV2:41*
use_locking(*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(

save/Assign_42Assignencoder/Variable_7/Adamsave/RestoreV2:42*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(*
use_locking(

save/Assign_43Assignencoder/Variable_7/Adam_1save/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@encoder/Variable_7*
validate_shape(
ø
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
:
loss_1/tagsConst*
valueB Bloss_1*
dtype0
9
loss_1ScalarSummaryloss_1/tags
loss/add_3*
T0
3
Merge/MergeSummaryMergeSummaryloss_1*
N
ê

initNoOp^beta1_power/Assign^beta2_power/Assign^decoder/Variable/Adam/Assign^decoder/Variable/Adam_1/Assign^decoder/Variable/Assign^decoder/Variable_1/Adam/Assign!^decoder/Variable_1/Adam_1/Assign^decoder/Variable_1/Assign^decoder/Variable_2/Adam/Assign!^decoder/Variable_2/Adam_1/Assign^decoder/Variable_2/Assign^decoder/Variable_3/Adam/Assign!^decoder/Variable_3/Adam_1/Assign^decoder/Variable_3/Assign^decoder/Variable_4/Adam/Assign!^decoder/Variable_4/Adam_1/Assign^decoder/Variable_4/Assign^decoder/Variable_5/Adam/Assign!^decoder/Variable_5/Adam_1/Assign^decoder/Variable_5/Assign^encoder/Variable/Adam/Assign^encoder/Variable/Adam_1/Assign^encoder/Variable/Assign^encoder/Variable_1/Adam/Assign!^encoder/Variable_1/Adam_1/Assign^encoder/Variable_1/Assign^encoder/Variable_2/Adam/Assign!^encoder/Variable_2/Adam_1/Assign^encoder/Variable_2/Assign^encoder/Variable_3/Adam/Assign!^encoder/Variable_3/Adam_1/Assign^encoder/Variable_3/Assign^encoder/Variable_4/Adam/Assign!^encoder/Variable_4/Adam_1/Assign^encoder/Variable_4/Assign^encoder/Variable_5/Adam/Assign!^encoder/Variable_5/Adam_1/Assign^encoder/Variable_5/Assign^encoder/Variable_6/Adam/Assign!^encoder/Variable_6/Adam_1/Assign^encoder/Variable_6/Assign^encoder/Variable_7/Adam/Assign!^encoder/Variable_7/Adam_1/Assign^encoder/Variable_7/Assign"