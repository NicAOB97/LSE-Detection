??:
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??6
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@ *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

: *
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	~?*,
shared_namelstm_12/lstm_cell_12/kernel
?
/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	~?*
dtype0
?
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
?
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel*
_output_shapes
:	@?*
dtype0
?
lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_12/lstm_cell_12/bias
?
-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
_output_shapes	
:?*
dtype0
?
lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*,
shared_namelstm_13/lstm_cell_13/kernel
?
/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel*
_output_shapes
:	@?*
dtype0
?
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
?
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_13/lstm_cell_13/bias
?
-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:?*
dtype0
?
lstm_14/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelstm_14/lstm_cell_14/kernel
?
/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/kernel* 
_output_shapes
:
??*
dtype0
?
%lstm_14/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*6
shared_name'%lstm_14/lstm_cell_14/recurrent_kernel
?
9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_14/recurrent_kernel*
_output_shapes
:	@?*
dtype0
?
lstm_14/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_14/lstm_cell_14/bias
?
-lstm_14/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	~?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m
?
6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	~?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m
?
4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m
?
6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m*
_output_shapes
:	@?*
dtype0
?
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
?
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m
?
4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_14/lstm_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/m
?
6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
?
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_14/lstm_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/m
?
4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	~?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v
?
6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	~?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v
?
4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v
?
6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v*
_output_shapes
:	@?*
dtype0
?
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
?
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v
?
4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_14/lstm_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/v
?
6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
?
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_14/lstm_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/v
?
4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?R
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Q
value?QB?Q B?Q
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
?
1iter

2beta_1

3beta_2
	4decay
5learning_ratem? m?%m?&m?+m?,m?6m?7m?8m?9m?:m?;m?<m?=m?>m?v? v?%v?&v?+v?,v?6v?7v?8v?9v?:v?;v?<v?=v?>v?
 
n
60
71
82
93
:4
;5
<6
=7
>8
9
 10
%11
&12
+13
,14
n
60
71
82
93
:4
;5
<6
=7
>8
9
 10
%11
&12
+13
,14
?
regularization_losses
?non_trainable_variables
@metrics
		variables
Alayer_metrics

trainable_variables

Blayers
Clayer_regularization_losses
 
~

6kernel
7recurrent_kernel
8bias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
 
 

60
71
82

60
71
82
?
regularization_losses
Hnon_trainable_variables

Istates
Jmetrics
	variables
Klayer_metrics
trainable_variables

Llayers
Mlayer_regularization_losses
~

9kernel
:recurrent_kernel
;bias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
 
 

90
:1
;2

90
:1
;2
?
regularization_losses
Rnon_trainable_variables

Sstates
Tmetrics
	variables
Ulayer_metrics
trainable_variables

Vlayers
Wlayer_regularization_losses
~

<kernel
=recurrent_kernel
>bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
 
 

<0
=1
>2

<0
=1
>2
?
regularization_losses
\non_trainable_variables

]states
^metrics
	variables
_layer_metrics
trainable_variables

`layers
alayer_regularization_losses
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
!regularization_losses
bnon_trainable_variables
cmetrics
"	variables
dlayer_metrics
#trainable_variables

elayers
flayer_regularization_losses
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
'regularization_losses
gnon_trainable_variables
hmetrics
(	variables
ilayer_metrics
)trainable_variables

jlayers
klayer_regularization_losses
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
-regularization_losses
lnon_trainable_variables
mmetrics
.	variables
nlayer_metrics
/trainable_variables

olayers
player_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_12/lstm_cell_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_12/lstm_cell_12/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_12/lstm_cell_12/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_13/lstm_cell_13/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_13/lstm_cell_13/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_13/lstm_cell_13/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_14/lstm_cell_14/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_14/lstm_cell_14/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_14/lstm_cell_14/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1
 
*
0
1
2
3
4
5
 
 

60
71
82

60
71
82
?
Dregularization_losses
snon_trainable_variables
tmetrics
E	variables
ulayer_metrics
Ftrainable_variables

vlayers
wlayer_regularization_losses
 
 
 
 

0
 
 

90
:1
;2

90
:1
;2
?
Nregularization_losses
xnon_trainable_variables
ymetrics
O	variables
zlayer_metrics
Ptrainable_variables

{layers
|layer_regularization_losses
 
 
 
 

0
 
 

<0
=1
>2

<0
=1
>2
?
Xregularization_losses
}non_trainable_variables
~metrics
Y	variables
layer_metrics
Ztrainable_variables
?layers
 ?layer_regularization_losses
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:?????????
~*
dtype0* 
shape:?????????
~
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biaslstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_89912
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOp9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOp-lstm_12/lstm_cell_12/bias/Read/ReadVariableOp/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_13/bias/Read/ReadVariableOp/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOp9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOp-lstm_14/lstm_cell_14/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpConst*C
Tin<
:28	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_93439
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biaslstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biastotalcounttotal_1count_1Adam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/m"Adam/lstm_12/lstm_cell_12/kernel/m,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m Adam/lstm_12/lstm_cell_12/bias/m"Adam/lstm_13/lstm_cell_13/kernel/m,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m Adam/lstm_13/lstm_cell_13/bias/m"Adam/lstm_14/lstm_cell_14/kernel/m,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m Adam/lstm_14/lstm_cell_14/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v"Adam/lstm_12/lstm_cell_12/kernel/v,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v Adam/lstm_12/lstm_cell_12/bias/v"Adam/lstm_13/lstm_cell_13/kernel/v,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v Adam/lstm_13/lstm_cell_13/bias/v"Adam/lstm_14/lstm_cell_14/kernel/v,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v Adam/lstm_14/lstm_cell_14/bias/v*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_93611??4
?
?
,__inference_lstm_cell_12_layer_call_fn_93037

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_91735
inputs_0/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91650*
condR
while_cond_91649*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
,__inference_lstm_cell_14_layer_call_fn_93254

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?
?
'__inference_lstm_14_layer_call_fn_92555

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_894112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?	
?
C__inference_dense_14_layer_call_and_return_conditional_losses_89659

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_89632

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_dense_14_layer_call_fn_92954

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_896592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
lstm_14_while_cond_90749,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1C
?lstm_14_while_lstm_14_while_cond_90749___redundant_placeholder0C
?lstm_14_while_lstm_14_while_cond_90749___redundant_placeholder1C
?lstm_14_while_lstm_14_while_cond_90749___redundant_placeholder2C
?lstm_14_while_lstm_14_while_cond_90749___redundant_placeholder3
lstm_14_while_identity
?
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_91147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_89143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_89143___redundant_placeholder03
/while_while_cond_89143___redundant_placeholder13
/while_while_cond_89143___redundant_placeholder23
/while_while_cond_89143___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_92786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92786___redundant_placeholder03
/while_while_cond_92786___redundant_placeholder13
/while_while_cond_92786___redundant_placeholder23
/while_while_cond_92786___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_12_layer_call_fn_93054

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_89564

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:
??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_89479*
condR
while_cond_89478*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?

?
,__inference_sequential_4_layer_call_fn_89867
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_898342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
?
?
while_cond_90993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_90993___redundant_placeholder03
/while_while_cond_90993___redundant_placeholder13
/while_while_cond_90993___redundant_placeholder23
/while_while_cond_90993___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?D
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_87834

inputs
lstm_cell_13_87752
lstm_cell_13_87754
lstm_cell_13_87756
identity??$lstm_cell_13/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_87752lstm_cell_13_87754lstm_cell_13_87756*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874382&
$lstm_cell_13/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_87752lstm_cell_13_87754lstm_cell_13_87756*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_87765*
condR
while_cond_87764*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0%^lstm_cell_13/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?O
?	
lstm_14_while_body_90750,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0A
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0@
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor=
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource?
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource>
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItem?
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_14/while/lstm_cell_14/MatMul?
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_14/while/lstm_cell_14/MatMul_1?
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_14/while/lstm_cell_14/add?
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_14/while/lstm_cell_14/BiasAdd?
 lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_14/while/lstm_cell_14/Const?
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim?
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_14/while/lstm_cell_14/split?
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_14/while/lstm_cell_14/Sigmoid?
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_1?
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_14/while/lstm_cell_14/mul?
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_14/while/lstm_cell_14/Relu?
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_1?
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/add_1?
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_2?
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_14/while/lstm_cell_14/Relu_1?
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_2?
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/y?
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/y?
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1?
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity?
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1?
lstm_14/while/Identity_2Identitylstm_14/while/add:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2?
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3?
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_4?
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_5"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_12_while_cond_90451,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1C
?lstm_12_while_lstm_12_while_cond_90451___redundant_placeholder0C
?lstm_12_while_lstm_12_while_cond_90451___redundant_placeholder1C
?lstm_12_while_lstm_12_while_cond_90451___redundant_placeholder2C
?lstm_12_while_lstm_12_while_cond_90451___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_14_layer_call_fn_92894
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_885762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92544

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:
??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_92459*
condR
while_cond_92458*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
}
(__inference_dense_12_layer_call_fn_92914

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_896052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_lstm_13_layer_call_fn_91899
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_878342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
??
?
!__inference__traced_restore_93611
file_prefix$
 assignvariableop_dense_12_kernel$
 assignvariableop_1_dense_12_bias&
"assignvariableop_2_dense_13_kernel$
 assignvariableop_3_dense_13_bias&
"assignvariableop_4_dense_14_kernel$
 assignvariableop_5_dense_14_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate3
/assignvariableop_11_lstm_12_lstm_cell_12_kernel=
9assignvariableop_12_lstm_12_lstm_cell_12_recurrent_kernel1
-assignvariableop_13_lstm_12_lstm_cell_12_bias3
/assignvariableop_14_lstm_13_lstm_cell_13_kernel=
9assignvariableop_15_lstm_13_lstm_cell_13_recurrent_kernel1
-assignvariableop_16_lstm_13_lstm_cell_13_bias3
/assignvariableop_17_lstm_14_lstm_cell_14_kernel=
9assignvariableop_18_lstm_14_lstm_cell_14_recurrent_kernel1
-assignvariableop_19_lstm_14_lstm_cell_14_bias
assignvariableop_20_total
assignvariableop_21_count
assignvariableop_22_total_1
assignvariableop_23_count_1.
*assignvariableop_24_adam_dense_12_kernel_m,
(assignvariableop_25_adam_dense_12_bias_m.
*assignvariableop_26_adam_dense_13_kernel_m,
(assignvariableop_27_adam_dense_13_bias_m.
*assignvariableop_28_adam_dense_14_kernel_m,
(assignvariableop_29_adam_dense_14_bias_m:
6assignvariableop_30_adam_lstm_12_lstm_cell_12_kernel_mD
@assignvariableop_31_adam_lstm_12_lstm_cell_12_recurrent_kernel_m8
4assignvariableop_32_adam_lstm_12_lstm_cell_12_bias_m:
6assignvariableop_33_adam_lstm_13_lstm_cell_13_kernel_mD
@assignvariableop_34_adam_lstm_13_lstm_cell_13_recurrent_kernel_m8
4assignvariableop_35_adam_lstm_13_lstm_cell_13_bias_m:
6assignvariableop_36_adam_lstm_14_lstm_cell_14_kernel_mD
@assignvariableop_37_adam_lstm_14_lstm_cell_14_recurrent_kernel_m8
4assignvariableop_38_adam_lstm_14_lstm_cell_14_bias_m.
*assignvariableop_39_adam_dense_12_kernel_v,
(assignvariableop_40_adam_dense_12_bias_v.
*assignvariableop_41_adam_dense_13_kernel_v,
(assignvariableop_42_adam_dense_13_bias_v.
*assignvariableop_43_adam_dense_14_kernel_v,
(assignvariableop_44_adam_dense_14_bias_v:
6assignvariableop_45_adam_lstm_12_lstm_cell_12_kernel_vD
@assignvariableop_46_adam_lstm_12_lstm_cell_12_recurrent_kernel_v8
4assignvariableop_47_adam_lstm_12_lstm_cell_12_bias_v:
6assignvariableop_48_adam_lstm_13_lstm_cell_13_kernel_vD
@assignvariableop_49_adam_lstm_13_lstm_cell_13_recurrent_kernel_v8
4assignvariableop_50_adam_lstm_13_lstm_cell_13_bias_v:
6assignvariableop_51_adam_lstm_14_lstm_cell_14_kernel_vD
@assignvariableop_52_adam_lstm_14_lstm_cell_14_recurrent_kernel_v8
4assignvariableop_53_adam_lstm_14_lstm_cell_14_bias_v
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_12_lstm_cell_12_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_lstm_12_lstm_cell_12_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_12_lstm_cell_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_lstm_13_lstm_cell_13_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp9assignvariableop_15_lstm_13_lstm_cell_13_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_lstm_13_lstm_cell_13_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_lstm_14_lstm_cell_14_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_lstm_14_lstm_cell_14_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_14_lstm_cell_14_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_12_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_12_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_13_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_13_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_14_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_14_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_12_lstm_cell_12_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_12_lstm_cell_12_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_13_lstm_cell_13_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_13_lstm_cell_13_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_lstm_14_lstm_cell_14_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp@assignvariableop_37_adam_lstm_14_lstm_cell_14_recurrent_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_lstm_14_lstm_cell_14_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_12_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_12_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_13_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_13_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_14_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_14_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_lstm_12_lstm_cell_12_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_lstm_12_lstm_cell_12_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_lstm_13_lstm_cell_13_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp@assignvariableop_49_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_lstm_13_lstm_cell_13_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_lstm_14_lstm_cell_14_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_lstm_14_lstm_cell_14_recurrent_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_lstm_14_lstm_cell_14_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_90856

inputs7
3lstm_12_lstm_cell_12_matmul_readvariableop_resource9
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource8
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource7
3lstm_13_lstm_cell_13_matmul_readvariableop_resource9
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource8
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource7
3lstm_14_lstm_cell_14_matmul_readvariableop_resource9
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource8
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?lstm_12/while?+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?lstm_13/while?+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?lstm_14/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAddz
lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/lstm_cell_12/Const?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_12_while_body_90452*$
condR
lstm_12_while_cond_90451*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtimee
lstm_13/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicem
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lesss
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessw
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zeros_1?
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm?
lstm_13/transpose	Transposelstm_12/transpose_1:y:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1?
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack?
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1?
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2?
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1?
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_13/TensorArrayV2/element_shape?
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor?
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack?
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1?
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2?
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_13/strided_slice_2?
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul?
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul_1?
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add?
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/BiasAddz
lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/lstm_cell_13/Const?
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim?
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_13/lstm_cell_13/split?
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Sigmoid?
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_1?
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul?
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu?
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_1?
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add_1?
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_2?
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu_1?
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_2?
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_13/TensorArrayV2_1/element_shape?
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time?
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter?
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_13_while_body_90601*$
condR
lstm_13_while_cond_90600*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_13/while?
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack?
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_13/strided_slice_3/stack?
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1?
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2?
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_3?
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm?
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtimee
lstm_14/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_14/Shape?
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack?
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1?
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2?
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicel
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/mul/y?
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros/Less/y?
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lessr
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/packed/1?
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/Const?
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zerosp
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/mul/y?
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros_1/Less/y?
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessv
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/packed/1?
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/Const?
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zeros_1?
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm?
lstm_14/transpose	Transposelstm_13/transpose_1:y:0lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:
??????????2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1?
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack?
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1?
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2?
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1?
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_14/TensorArrayV2/element_shape?
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor?
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack?
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1?
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2?
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_14/strided_slice_2?
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul?
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul_1?
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/add?
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/BiasAddz
lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/lstm_cell_14/Const?
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim?
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_14/lstm_cell_14/split?
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Sigmoid?
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_1?
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul?
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu?
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_1?
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/add_1?
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_2?
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu_1?
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_2?
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_14/TensorArrayV2_1/element_shape?
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/time?
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter?
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_14_while_body_90750*$
condR
lstm_14_while_cond_90749*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_14/while?
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack?
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_14/strided_slice_3/stack?
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1?
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2?
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_14/strided_slice_3?
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm?
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtime?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul lstm_14/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_13/Relu?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_14/Softmax?
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91560

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91475*
condR
while_cond_91474*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_89229

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_89144*
condR
while_cond_89143*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?

?
lstm_13_while_cond_90600,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1C
?lstm_13_while_lstm_13_while_cond_90600___redundant_placeholder0C
?lstm_13_while_lstm_13_while_cond_90600___redundant_placeholder1C
?lstm_13_while_lstm_13_while_cond_90600___redundant_placeholder2C
?lstm_13_while_lstm_13_while_cond_90600___redundant_placeholder3
lstm_13_while_identity
?
lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?O
?	
lstm_12_while_body_90452,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0A
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0@
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor=
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource?
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource>
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
 lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_12/while/lstm_cell_12/Const?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1?
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1?
lstm_12/while/Identity_2Identitylstm_12/while/add:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_5"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_91474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91474___redundant_placeholder03
/while_while_cond_91474___redundant_placeholder13
/while_while_cond_91474___redundant_placeholder23
/while_while_cond_91474___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_91146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91146___redundant_placeholder03
/while_while_cond_91146___redundant_placeholder13
/while_while_cond_91146___redundant_placeholder23
/while_while_cond_91146___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92391

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:
??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_92306*
condR
while_cond_92305*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?C
?
while_body_91803
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_lstm_13_layer_call_fn_92238

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_892292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?!
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89676
lstm_12_input
lstm_12_88917
lstm_12_88919
lstm_12_88921
lstm_13_89252
lstm_13_89254
lstm_13_89256
lstm_14_89587
lstm_14_89589
lstm_14_89591
dense_12_89616
dense_12_89618
dense_13_89643
dense_13_89645
dense_14_89670
dense_14_89672
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_88917lstm_12_88919lstm_12_88921*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_887412!
lstm_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0lstm_13_89252lstm_13_89254lstm_13_89256*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_890762!
lstm_13/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0lstm_14_89587lstm_14_89589lstm_14_89591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_894112!
lstm_14/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_12_89616dense_12_89618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_896052"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_89643dense_13_89645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_896322"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_89670dense_14_89672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_896592"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
?
?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_93020

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?

?
,__inference_sequential_4_layer_call_fn_89792
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_897592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
??
?
 __inference__wrapped_model_86755
lstm_12_inputD
@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resourceF
Bsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceE
Asequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resourceD
@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resourceF
Bsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceE
Asequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resourceD
@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resourceF
Bsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceE
Asequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource8
4sequential_4_dense_12_matmul_readvariableop_resource9
5sequential_4_dense_12_biasadd_readvariableop_resource8
4sequential_4_dense_13_matmul_readvariableop_resource9
5sequential_4_dense_13_biasadd_readvariableop_resource8
4sequential_4_dense_14_matmul_readvariableop_resource9
5sequential_4_dense_14_biasadd_readvariableop_resource
identity??,sequential_4/dense_12/BiasAdd/ReadVariableOp?+sequential_4/dense_12/MatMul/ReadVariableOp?,sequential_4/dense_13/BiasAdd/ReadVariableOp?+sequential_4/dense_13/MatMul/ReadVariableOp?,sequential_4/dense_14/BiasAdd/ReadVariableOp?+sequential_4/dense_14/MatMul/ReadVariableOp?8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp?9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?sequential_4/lstm_12/while?8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp?9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?sequential_4/lstm_13/while?8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp?9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?sequential_4/lstm_14/whileu
sequential_4/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_4/lstm_12/Shape?
(sequential_4/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/lstm_12/strided_slice/stack?
*sequential_4/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_12/strided_slice/stack_1?
*sequential_4/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_12/strided_slice/stack_2?
"sequential_4/lstm_12/strided_sliceStridedSlice#sequential_4/lstm_12/Shape:output:01sequential_4/lstm_12/strided_slice/stack:output:03sequential_4/lstm_12/strided_slice/stack_1:output:03sequential_4/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_4/lstm_12/strided_slice?
 sequential_4/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_4/lstm_12/zeros/mul/y?
sequential_4/lstm_12/zeros/mulMul+sequential_4/lstm_12/strided_slice:output:0)sequential_4/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_12/zeros/mul?
!sequential_4/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_4/lstm_12/zeros/Less/y?
sequential_4/lstm_12/zeros/LessLess"sequential_4/lstm_12/zeros/mul:z:0*sequential_4/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_12/zeros/Less?
#sequential_4/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_4/lstm_12/zeros/packed/1?
!sequential_4/lstm_12/zeros/packedPack+sequential_4/lstm_12/strided_slice:output:0,sequential_4/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_4/lstm_12/zeros/packed?
 sequential_4/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_4/lstm_12/zeros/Const?
sequential_4/lstm_12/zerosFill*sequential_4/lstm_12/zeros/packed:output:0)sequential_4/lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_4/lstm_12/zeros?
"sequential_4/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_4/lstm_12/zeros_1/mul/y?
 sequential_4/lstm_12/zeros_1/mulMul+sequential_4/lstm_12/strided_slice:output:0+sequential_4/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_12/zeros_1/mul?
#sequential_4/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_4/lstm_12/zeros_1/Less/y?
!sequential_4/lstm_12/zeros_1/LessLess$sequential_4/lstm_12/zeros_1/mul:z:0,sequential_4/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_4/lstm_12/zeros_1/Less?
%sequential_4/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_4/lstm_12/zeros_1/packed/1?
#sequential_4/lstm_12/zeros_1/packedPack+sequential_4/lstm_12/strided_slice:output:0.sequential_4/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/lstm_12/zeros_1/packed?
"sequential_4/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_4/lstm_12/zeros_1/Const?
sequential_4/lstm_12/zeros_1Fill,sequential_4/lstm_12/zeros_1/packed:output:0+sequential_4/lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_4/lstm_12/zeros_1?
#sequential_4/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_4/lstm_12/transpose/perm?
sequential_4/lstm_12/transpose	Transposelstm_12_input,sequential_4/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????~2 
sequential_4/lstm_12/transpose?
sequential_4/lstm_12/Shape_1Shape"sequential_4/lstm_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_12/Shape_1?
*sequential_4/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_12/strided_slice_1/stack?
,sequential_4/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_12/strided_slice_1/stack_1?
,sequential_4/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_12/strided_slice_1/stack_2?
$sequential_4/lstm_12/strided_slice_1StridedSlice%sequential_4/lstm_12/Shape_1:output:03sequential_4/lstm_12/strided_slice_1/stack:output:05sequential_4/lstm_12/strided_slice_1/stack_1:output:05sequential_4/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/lstm_12/strided_slice_1?
0sequential_4/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_4/lstm_12/TensorArrayV2/element_shape?
"sequential_4/lstm_12/TensorArrayV2TensorListReserve9sequential_4/lstm_12/TensorArrayV2/element_shape:output:0-sequential_4/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_4/lstm_12/TensorArrayV2?
Jsequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2L
Jsequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_12/transpose:y:0Ssequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor?
*sequential_4/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_12/strided_slice_2/stack?
,sequential_4/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_12/strided_slice_2/stack_1?
,sequential_4/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_12/strided_slice_2/stack_2?
$sequential_4/lstm_12/strided_slice_2StridedSlice"sequential_4/lstm_12/transpose:y:03sequential_4/lstm_12/strided_slice_2/stack:output:05sequential_4/lstm_12/strided_slice_2/stack_1:output:05sequential_4/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2&
$sequential_4/lstm_12/strided_slice_2?
7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype029
7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
(sequential_4/lstm_12/lstm_cell_12/MatMulMatMul-sequential_4/lstm_12/strided_slice_2:output:0?sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_4/lstm_12/lstm_cell_12/MatMul?
9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02;
9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
*sequential_4/lstm_12/lstm_cell_12/MatMul_1MatMul#sequential_4/lstm_12/zeros:output:0Asequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_4/lstm_12/lstm_cell_12/MatMul_1?
%sequential_4/lstm_12/lstm_cell_12/addAddV22sequential_4/lstm_12/lstm_cell_12/MatMul:product:04sequential_4/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_12/lstm_cell_12/add?
8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
)sequential_4/lstm_12/lstm_cell_12/BiasAddBiasAdd)sequential_4/lstm_12/lstm_cell_12/add:z:0@sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_4/lstm_12/lstm_cell_12/BiasAdd?
'sequential_4/lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_4/lstm_12/lstm_cell_12/Const?
1sequential_4/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/lstm_12/lstm_cell_12/split/split_dim?
'sequential_4/lstm_12/lstm_cell_12/splitSplit:sequential_4/lstm_12/lstm_cell_12/split/split_dim:output:02sequential_4/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2)
'sequential_4/lstm_12/lstm_cell_12/split?
)sequential_4/lstm_12/lstm_cell_12/SigmoidSigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2+
)sequential_4/lstm_12/lstm_cell_12/Sigmoid?
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_1?
%sequential_4/lstm_12/lstm_cell_12/mulMul/sequential_4/lstm_12/lstm_cell_12/Sigmoid_1:y:0%sequential_4/lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_12/lstm_cell_12/mul?
&sequential_4/lstm_12/lstm_cell_12/ReluRelu0sequential_4/lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2(
&sequential_4/lstm_12/lstm_cell_12/Relu?
'sequential_4/lstm_12/lstm_cell_12/mul_1Mul-sequential_4/lstm_12/lstm_cell_12/Sigmoid:y:04sequential_4/lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_12/lstm_cell_12/mul_1?
'sequential_4/lstm_12/lstm_cell_12/add_1AddV2)sequential_4/lstm_12/lstm_cell_12/mul:z:0+sequential_4/lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_12/lstm_cell_12/add_1?
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_2?
(sequential_4/lstm_12/lstm_cell_12/Relu_1Relu+sequential_4/lstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2*
(sequential_4/lstm_12/lstm_cell_12/Relu_1?
'sequential_4/lstm_12/lstm_cell_12/mul_2Mul/sequential_4/lstm_12/lstm_cell_12/Sigmoid_2:y:06sequential_4/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_12/lstm_cell_12/mul_2?
2sequential_4/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   24
2sequential_4/lstm_12/TensorArrayV2_1/element_shape?
$sequential_4/lstm_12/TensorArrayV2_1TensorListReserve;sequential_4/lstm_12/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_4/lstm_12/TensorArrayV2_1x
sequential_4/lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_12/time?
-sequential_4/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_4/lstm_12/while/maximum_iterations?
'sequential_4/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_4/lstm_12/while/loop_counter?
sequential_4/lstm_12/whileWhile0sequential_4/lstm_12/while/loop_counter:output:06sequential_4/lstm_12/while/maximum_iterations:output:0"sequential_4/lstm_12/time:output:0-sequential_4/lstm_12/TensorArrayV2_1:handle:0#sequential_4/lstm_12/zeros:output:0%sequential_4/lstm_12/zeros_1:output:0-sequential_4/lstm_12/strided_slice_1:output:0Lsequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resourceBsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceAsequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_4_lstm_12_while_body_86351*1
cond)R'
%sequential_4_lstm_12_while_cond_86350*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
sequential_4/lstm_12/while?
Esequential_4/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2G
Esequential_4/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_4/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_12/while:output:3Nsequential_4/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype029
7sequential_4/lstm_12/TensorArrayV2Stack/TensorListStack?
*sequential_4/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_4/lstm_12/strided_slice_3/stack?
,sequential_4/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_4/lstm_12/strided_slice_3/stack_1?
,sequential_4/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_12/strided_slice_3/stack_2?
$sequential_4/lstm_12/strided_slice_3StridedSlice@sequential_4/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_12/strided_slice_3/stack:output:05sequential_4/lstm_12/strided_slice_3/stack_1:output:05sequential_4/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2&
$sequential_4/lstm_12/strided_slice_3?
%sequential_4/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_4/lstm_12/transpose_1/perm?
 sequential_4/lstm_12/transpose_1	Transpose@sequential_4/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2"
 sequential_4/lstm_12/transpose_1?
sequential_4/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_12/runtime?
sequential_4/lstm_13/ShapeShape$sequential_4/lstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_13/Shape?
(sequential_4/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/lstm_13/strided_slice/stack?
*sequential_4/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_13/strided_slice/stack_1?
*sequential_4/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_13/strided_slice/stack_2?
"sequential_4/lstm_13/strided_sliceStridedSlice#sequential_4/lstm_13/Shape:output:01sequential_4/lstm_13/strided_slice/stack:output:03sequential_4/lstm_13/strided_slice/stack_1:output:03sequential_4/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_4/lstm_13/strided_slice?
 sequential_4/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_4/lstm_13/zeros/mul/y?
sequential_4/lstm_13/zeros/mulMul+sequential_4/lstm_13/strided_slice:output:0)sequential_4/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_13/zeros/mul?
!sequential_4/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_4/lstm_13/zeros/Less/y?
sequential_4/lstm_13/zeros/LessLess"sequential_4/lstm_13/zeros/mul:z:0*sequential_4/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_13/zeros/Less?
#sequential_4/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_4/lstm_13/zeros/packed/1?
!sequential_4/lstm_13/zeros/packedPack+sequential_4/lstm_13/strided_slice:output:0,sequential_4/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_4/lstm_13/zeros/packed?
 sequential_4/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_4/lstm_13/zeros/Const?
sequential_4/lstm_13/zerosFill*sequential_4/lstm_13/zeros/packed:output:0)sequential_4/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_4/lstm_13/zeros?
"sequential_4/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_4/lstm_13/zeros_1/mul/y?
 sequential_4/lstm_13/zeros_1/mulMul+sequential_4/lstm_13/strided_slice:output:0+sequential_4/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_13/zeros_1/mul?
#sequential_4/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_4/lstm_13/zeros_1/Less/y?
!sequential_4/lstm_13/zeros_1/LessLess$sequential_4/lstm_13/zeros_1/mul:z:0,sequential_4/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_4/lstm_13/zeros_1/Less?
%sequential_4/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_4/lstm_13/zeros_1/packed/1?
#sequential_4/lstm_13/zeros_1/packedPack+sequential_4/lstm_13/strided_slice:output:0.sequential_4/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/lstm_13/zeros_1/packed?
"sequential_4/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_4/lstm_13/zeros_1/Const?
sequential_4/lstm_13/zeros_1Fill,sequential_4/lstm_13/zeros_1/packed:output:0+sequential_4/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_4/lstm_13/zeros_1?
#sequential_4/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_4/lstm_13/transpose/perm?
sequential_4/lstm_13/transpose	Transpose$sequential_4/lstm_12/transpose_1:y:0,sequential_4/lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????@2 
sequential_4/lstm_13/transpose?
sequential_4/lstm_13/Shape_1Shape"sequential_4/lstm_13/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_13/Shape_1?
*sequential_4/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_13/strided_slice_1/stack?
,sequential_4/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_13/strided_slice_1/stack_1?
,sequential_4/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_13/strided_slice_1/stack_2?
$sequential_4/lstm_13/strided_slice_1StridedSlice%sequential_4/lstm_13/Shape_1:output:03sequential_4/lstm_13/strided_slice_1/stack:output:05sequential_4/lstm_13/strided_slice_1/stack_1:output:05sequential_4/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/lstm_13/strided_slice_1?
0sequential_4/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_4/lstm_13/TensorArrayV2/element_shape?
"sequential_4/lstm_13/TensorArrayV2TensorListReserve9sequential_4/lstm_13/TensorArrayV2/element_shape:output:0-sequential_4/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_4/lstm_13/TensorArrayV2?
Jsequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2L
Jsequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_13/transpose:y:0Ssequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor?
*sequential_4/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_13/strided_slice_2/stack?
,sequential_4/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_13/strided_slice_2/stack_1?
,sequential_4/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_13/strided_slice_2/stack_2?
$sequential_4/lstm_13/strided_slice_2StridedSlice"sequential_4/lstm_13/transpose:y:03sequential_4/lstm_13/strided_slice_2/stack:output:05sequential_4/lstm_13/strided_slice_2/stack_1:output:05sequential_4/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2&
$sequential_4/lstm_13/strided_slice_2?
7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype029
7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
(sequential_4/lstm_13/lstm_cell_13/MatMulMatMul-sequential_4/lstm_13/strided_slice_2:output:0?sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_4/lstm_13/lstm_cell_13/MatMul?
9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
*sequential_4/lstm_13/lstm_cell_13/MatMul_1MatMul#sequential_4/lstm_13/zeros:output:0Asequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_4/lstm_13/lstm_cell_13/MatMul_1?
%sequential_4/lstm_13/lstm_cell_13/addAddV22sequential_4/lstm_13/lstm_cell_13/MatMul:product:04sequential_4/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_13/lstm_cell_13/add?
8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
)sequential_4/lstm_13/lstm_cell_13/BiasAddBiasAdd)sequential_4/lstm_13/lstm_cell_13/add:z:0@sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_4/lstm_13/lstm_cell_13/BiasAdd?
'sequential_4/lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_4/lstm_13/lstm_cell_13/Const?
1sequential_4/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/lstm_13/lstm_cell_13/split/split_dim?
'sequential_4/lstm_13/lstm_cell_13/splitSplit:sequential_4/lstm_13/lstm_cell_13/split/split_dim:output:02sequential_4/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2)
'sequential_4/lstm_13/lstm_cell_13/split?
)sequential_4/lstm_13/lstm_cell_13/SigmoidSigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_4/lstm_13/lstm_cell_13/Sigmoid?
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_1?
%sequential_4/lstm_13/lstm_cell_13/mulMul/sequential_4/lstm_13/lstm_cell_13/Sigmoid_1:y:0%sequential_4/lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_13/lstm_cell_13/mul?
&sequential_4/lstm_13/lstm_cell_13/ReluRelu0sequential_4/lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2(
&sequential_4/lstm_13/lstm_cell_13/Relu?
'sequential_4/lstm_13/lstm_cell_13/mul_1Mul-sequential_4/lstm_13/lstm_cell_13/Sigmoid:y:04sequential_4/lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_4/lstm_13/lstm_cell_13/mul_1?
'sequential_4/lstm_13/lstm_cell_13/add_1AddV2)sequential_4/lstm_13/lstm_cell_13/mul:z:0+sequential_4/lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_4/lstm_13/lstm_cell_13/add_1?
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_2?
(sequential_4/lstm_13/lstm_cell_13/Relu_1Relu+sequential_4/lstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_4/lstm_13/lstm_cell_13/Relu_1?
'sequential_4/lstm_13/lstm_cell_13/mul_2Mul/sequential_4/lstm_13/lstm_cell_13/Sigmoid_2:y:06sequential_4/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_4/lstm_13/lstm_cell_13/mul_2?
2sequential_4/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_4/lstm_13/TensorArrayV2_1/element_shape?
$sequential_4/lstm_13/TensorArrayV2_1TensorListReserve;sequential_4/lstm_13/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_4/lstm_13/TensorArrayV2_1x
sequential_4/lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_13/time?
-sequential_4/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_4/lstm_13/while/maximum_iterations?
'sequential_4/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_4/lstm_13/while/loop_counter?
sequential_4/lstm_13/whileWhile0sequential_4/lstm_13/while/loop_counter:output:06sequential_4/lstm_13/while/maximum_iterations:output:0"sequential_4/lstm_13/time:output:0-sequential_4/lstm_13/TensorArrayV2_1:handle:0#sequential_4/lstm_13/zeros:output:0%sequential_4/lstm_13/zeros_1:output:0-sequential_4/lstm_13/strided_slice_1:output:0Lsequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resourceBsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceAsequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_4_lstm_13_while_body_86500*1
cond)R'
%sequential_4_lstm_13_while_cond_86499*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_4/lstm_13/while?
Esequential_4/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_4/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_4/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_13/while:output:3Nsequential_4/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype029
7sequential_4/lstm_13/TensorArrayV2Stack/TensorListStack?
*sequential_4/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_4/lstm_13/strided_slice_3/stack?
,sequential_4/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_4/lstm_13/strided_slice_3/stack_1?
,sequential_4/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_13/strided_slice_3/stack_2?
$sequential_4/lstm_13/strided_slice_3StridedSlice@sequential_4/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_13/strided_slice_3/stack:output:05sequential_4/lstm_13/strided_slice_3/stack_1:output:05sequential_4/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_4/lstm_13/strided_slice_3?
%sequential_4/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_4/lstm_13/transpose_1/perm?
 sequential_4/lstm_13/transpose_1	Transpose@sequential_4/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2"
 sequential_4/lstm_13/transpose_1?
sequential_4/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_13/runtime?
sequential_4/lstm_14/ShapeShape$sequential_4/lstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_14/Shape?
(sequential_4/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_4/lstm_14/strided_slice/stack?
*sequential_4/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_14/strided_slice/stack_1?
*sequential_4/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_4/lstm_14/strided_slice/stack_2?
"sequential_4/lstm_14/strided_sliceStridedSlice#sequential_4/lstm_14/Shape:output:01sequential_4/lstm_14/strided_slice/stack:output:03sequential_4/lstm_14/strided_slice/stack_1:output:03sequential_4/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_4/lstm_14/strided_slice?
 sequential_4/lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_4/lstm_14/zeros/mul/y?
sequential_4/lstm_14/zeros/mulMul+sequential_4/lstm_14/strided_slice:output:0)sequential_4/lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_14/zeros/mul?
!sequential_4/lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_4/lstm_14/zeros/Less/y?
sequential_4/lstm_14/zeros/LessLess"sequential_4/lstm_14/zeros/mul:z:0*sequential_4/lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_14/zeros/Less?
#sequential_4/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_4/lstm_14/zeros/packed/1?
!sequential_4/lstm_14/zeros/packedPack+sequential_4/lstm_14/strided_slice:output:0,sequential_4/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_4/lstm_14/zeros/packed?
 sequential_4/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_4/lstm_14/zeros/Const?
sequential_4/lstm_14/zerosFill*sequential_4/lstm_14/zeros/packed:output:0)sequential_4/lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_4/lstm_14/zeros?
"sequential_4/lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_4/lstm_14/zeros_1/mul/y?
 sequential_4/lstm_14/zeros_1/mulMul+sequential_4/lstm_14/strided_slice:output:0+sequential_4/lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_14/zeros_1/mul?
#sequential_4/lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_4/lstm_14/zeros_1/Less/y?
!sequential_4/lstm_14/zeros_1/LessLess$sequential_4/lstm_14/zeros_1/mul:z:0,sequential_4/lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_4/lstm_14/zeros_1/Less?
%sequential_4/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_4/lstm_14/zeros_1/packed/1?
#sequential_4/lstm_14/zeros_1/packedPack+sequential_4/lstm_14/strided_slice:output:0.sequential_4/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/lstm_14/zeros_1/packed?
"sequential_4/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_4/lstm_14/zeros_1/Const?
sequential_4/lstm_14/zeros_1Fill,sequential_4/lstm_14/zeros_1/packed:output:0+sequential_4/lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_4/lstm_14/zeros_1?
#sequential_4/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_4/lstm_14/transpose/perm?
sequential_4/lstm_14/transpose	Transpose$sequential_4/lstm_13/transpose_1:y:0,sequential_4/lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:
??????????2 
sequential_4/lstm_14/transpose?
sequential_4/lstm_14/Shape_1Shape"sequential_4/lstm_14/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_14/Shape_1?
*sequential_4/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_14/strided_slice_1/stack?
,sequential_4/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_14/strided_slice_1/stack_1?
,sequential_4/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_14/strided_slice_1/stack_2?
$sequential_4/lstm_14/strided_slice_1StridedSlice%sequential_4/lstm_14/Shape_1:output:03sequential_4/lstm_14/strided_slice_1/stack:output:05sequential_4/lstm_14/strided_slice_1/stack_1:output:05sequential_4/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/lstm_14/strided_slice_1?
0sequential_4/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_4/lstm_14/TensorArrayV2/element_shape?
"sequential_4/lstm_14/TensorArrayV2TensorListReserve9sequential_4/lstm_14/TensorArrayV2/element_shape:output:0-sequential_4/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_4/lstm_14/TensorArrayV2?
Jsequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_14/transpose:y:0Ssequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor?
*sequential_4/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/lstm_14/strided_slice_2/stack?
,sequential_4/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_14/strided_slice_2/stack_1?
,sequential_4/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_14/strided_slice_2/stack_2?
$sequential_4/lstm_14/strided_slice_2StridedSlice"sequential_4/lstm_14/transpose:y:03sequential_4/lstm_14/strided_slice_2/stack:output:05sequential_4/lstm_14/strided_slice_2/stack_1:output:05sequential_4/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_4/lstm_14/strided_slice_2?
7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
(sequential_4/lstm_14/lstm_cell_14/MatMulMatMul-sequential_4/lstm_14/strided_slice_2:output:0?sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_4/lstm_14/lstm_cell_14/MatMul?
9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02;
9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
*sequential_4/lstm_14/lstm_cell_14/MatMul_1MatMul#sequential_4/lstm_14/zeros:output:0Asequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_4/lstm_14/lstm_cell_14/MatMul_1?
%sequential_4/lstm_14/lstm_cell_14/addAddV22sequential_4/lstm_14/lstm_cell_14/MatMul:product:04sequential_4/lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_14/lstm_cell_14/add?
8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
)sequential_4/lstm_14/lstm_cell_14/BiasAddBiasAdd)sequential_4/lstm_14/lstm_cell_14/add:z:0@sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_4/lstm_14/lstm_cell_14/BiasAdd?
'sequential_4/lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_4/lstm_14/lstm_cell_14/Const?
1sequential_4/lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/lstm_14/lstm_cell_14/split/split_dim?
'sequential_4/lstm_14/lstm_cell_14/splitSplit:sequential_4/lstm_14/lstm_cell_14/split/split_dim:output:02sequential_4/lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2)
'sequential_4/lstm_14/lstm_cell_14/split?
)sequential_4/lstm_14/lstm_cell_14/SigmoidSigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2+
)sequential_4/lstm_14/lstm_cell_14/Sigmoid?
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_1Sigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_1?
%sequential_4/lstm_14/lstm_cell_14/mulMul/sequential_4/lstm_14/lstm_cell_14/Sigmoid_1:y:0%sequential_4/lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_14/lstm_cell_14/mul?
&sequential_4/lstm_14/lstm_cell_14/ReluRelu0sequential_4/lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2(
&sequential_4/lstm_14/lstm_cell_14/Relu?
'sequential_4/lstm_14/lstm_cell_14/mul_1Mul-sequential_4/lstm_14/lstm_cell_14/Sigmoid:y:04sequential_4/lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_14/lstm_cell_14/mul_1?
'sequential_4/lstm_14/lstm_cell_14/add_1AddV2)sequential_4/lstm_14/lstm_cell_14/mul:z:0+sequential_4/lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_14/lstm_cell_14/add_1?
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_2Sigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_2?
(sequential_4/lstm_14/lstm_cell_14/Relu_1Relu+sequential_4/lstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2*
(sequential_4/lstm_14/lstm_cell_14/Relu_1?
'sequential_4/lstm_14/lstm_cell_14/mul_2Mul/sequential_4/lstm_14/lstm_cell_14/Sigmoid_2:y:06sequential_4/lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2)
'sequential_4/lstm_14/lstm_cell_14/mul_2?
2sequential_4/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   24
2sequential_4/lstm_14/TensorArrayV2_1/element_shape?
$sequential_4/lstm_14/TensorArrayV2_1TensorListReserve;sequential_4/lstm_14/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_4/lstm_14/TensorArrayV2_1x
sequential_4/lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_14/time?
-sequential_4/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_4/lstm_14/while/maximum_iterations?
'sequential_4/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_4/lstm_14/while/loop_counter?
sequential_4/lstm_14/whileWhile0sequential_4/lstm_14/while/loop_counter:output:06sequential_4/lstm_14/while/maximum_iterations:output:0"sequential_4/lstm_14/time:output:0-sequential_4/lstm_14/TensorArrayV2_1:handle:0#sequential_4/lstm_14/zeros:output:0%sequential_4/lstm_14/zeros_1:output:0-sequential_4/lstm_14/strided_slice_1:output:0Lsequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resourceBsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceAsequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_4_lstm_14_while_body_86649*1
cond)R'
%sequential_4_lstm_14_while_cond_86648*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
sequential_4/lstm_14/while?
Esequential_4/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2G
Esequential_4/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_4/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_14/while:output:3Nsequential_4/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype029
7sequential_4/lstm_14/TensorArrayV2Stack/TensorListStack?
*sequential_4/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_4/lstm_14/strided_slice_3/stack?
,sequential_4/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_4/lstm_14/strided_slice_3/stack_1?
,sequential_4/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/lstm_14/strided_slice_3/stack_2?
$sequential_4/lstm_14/strided_slice_3StridedSlice@sequential_4/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_14/strided_slice_3/stack:output:05sequential_4/lstm_14/strided_slice_3/stack_1:output:05sequential_4/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2&
$sequential_4/lstm_14/strided_slice_3?
%sequential_4/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_4/lstm_14/transpose_1/perm?
 sequential_4/lstm_14/transpose_1	Transpose@sequential_4/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2"
 sequential_4/lstm_14/transpose_1?
sequential_4/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_14/runtime?
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp?
sequential_4/dense_12/MatMulMatMul-sequential_4/lstm_14/strided_slice_3:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_4/dense_12/MatMul?
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp?
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_4/dense_12/BiasAdd?
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_4/dense_12/Relu?
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp?
sequential_4/dense_13/MatMulMatMul(sequential_4/dense_12/Relu:activations:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_4/dense_13/MatMul?
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp?
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_4/dense_13/BiasAdd?
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_4/dense_13/Relu?
+sequential_4/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_4/dense_14/MatMul/ReadVariableOp?
sequential_4/dense_14/MatMulMatMul(sequential_4/dense_13/Relu:activations:03sequential_4/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_14/MatMul?
,sequential_4/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/dense_14/BiasAdd/ReadVariableOp?
sequential_4/dense_14/BiasAddBiasAdd&sequential_4/dense_14/MatMul:product:04sequential_4/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_14/BiasAdd?
sequential_4/dense_14/SoftmaxSoftmax&sequential_4/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_14/Softmax?
IdentityIdentity'sequential_4/dense_14/Softmax:softmax:0-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp-^sequential_4/dense_14/BiasAdd/ReadVariableOp,^sequential_4/dense_14/MatMul/ReadVariableOp9^sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8^sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:^sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^sequential_4/lstm_12/while9^sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8^sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:^sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^sequential_4/lstm_13/while9^sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8^sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:^sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^sequential_4/lstm_14/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp2\
,sequential_4/dense_14/BiasAdd/ReadVariableOp,sequential_4/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_14/MatMul/ReadVariableOp+sequential_4/dense_14/MatMul/ReadVariableOp2t
8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp2v
9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp28
sequential_4/lstm_12/whilesequential_4/lstm_12/while2t
8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp2v
9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp28
sequential_4/lstm_13/whilesequential_4/lstm_13/while2t
8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp2v
9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp28
sequential_4/lstm_14/whilesequential_4/lstm_14/while:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
?O
?	
lstm_13_while_body_90129,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0A
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0@
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor=
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource?
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource>
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource??1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItem?
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/MatMul?
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_13/while/lstm_cell_13/MatMul_1?
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/add?
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/BiasAdd?
 lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_13/while/lstm_cell_13/Const?
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim?
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_13/while/lstm_cell_13/split?
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/Sigmoid?
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_1?
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/mul?
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_13/while/lstm_cell_13/Relu?
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_1?
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/add_1?
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_2?
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/Relu_1?
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_2?
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y?
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y?
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1?
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity?
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1?
lstm_13/while/Identity_2Identitylstm_13/while/add:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2?
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3?
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_4?
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_5"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?C
?
while_body_88656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?O
?	
lstm_14_while_body_90278,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0A
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0@
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor=
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource?
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource>
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItem?
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_14/while/lstm_cell_14/MatMul?
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_14/while/lstm_cell_14/MatMul_1?
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_14/while/lstm_cell_14/add?
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_14/while/lstm_cell_14/BiasAdd?
 lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_14/while/lstm_cell_14/Const?
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim?
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_14/while/lstm_cell_14/split?
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_14/while/lstm_cell_14/Sigmoid?
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_1?
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_14/while/lstm_cell_14/mul?
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_14/while/lstm_cell_14/Relu?
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_1?
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/add_1?
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_2?
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_14/while/lstm_cell_14/Relu_1?
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_2?
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/y?
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/y?
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1?
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity?
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1?
lstm_14/while/Identity_2Identitylstm_14/while/add:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2?
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3?
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_4?
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:02^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_5"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_91802
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91802___redundant_placeholder03
/while_while_cond_91802___redundant_placeholder13
/while_while_cond_91802___redundant_placeholder23
/while_while_cond_91802___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_90384

inputs7
3lstm_12_lstm_cell_12_matmul_readvariableop_resource9
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource8
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource7
3lstm_13_lstm_cell_13_matmul_readvariableop_resource9
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource8
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource7
3lstm_14_lstm_cell_14_matmul_readvariableop_resource9
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource8
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?lstm_12/while?+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?lstm_13/while?+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?lstm_14/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAddz
lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/lstm_cell_12/Const?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_12_while_body_89980*$
condR
lstm_12_while_cond_89979*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtimee
lstm_13/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicem
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lesss
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessw
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zeros_1?
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm?
lstm_13/transpose	Transposelstm_12/transpose_1:y:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1?
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack?
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1?
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2?
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1?
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_13/TensorArrayV2/element_shape?
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor?
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack?
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1?
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2?
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_13/strided_slice_2?
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul?
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul_1?
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add?
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/BiasAddz
lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/lstm_cell_13/Const?
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim?
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_13/lstm_cell_13/split?
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Sigmoid?
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_1?
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul?
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu?
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_1?
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add_1?
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_2?
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu_1?
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_2?
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_13/TensorArrayV2_1/element_shape?
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time?
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter?
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_13_while_body_90129*$
condR
lstm_13_while_cond_90128*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_13/while?
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack?
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_13/strided_slice_3/stack?
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1?
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2?
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_3?
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm?
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtimee
lstm_14/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_14/Shape?
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack?
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1?
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2?
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicel
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/mul/y?
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros/Less/y?
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lessr
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/packed/1?
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/Const?
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zerosp
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/mul/y?
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros_1/Less/y?
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessv
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/packed/1?
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/Const?
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zeros_1?
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm?
lstm_14/transpose	Transposelstm_13/transpose_1:y:0lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:
??????????2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1?
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack?
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1?
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2?
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1?
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_14/TensorArrayV2/element_shape?
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor?
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack?
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1?
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2?
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_14/strided_slice_2?
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul?
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul_1?
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/add?
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/BiasAddz
lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/lstm_cell_14/Const?
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim?
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_14/lstm_cell_14/split?
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Sigmoid?
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_1?
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul?
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu?
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_1?
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/add_1?
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_2?
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu_1?
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_2?
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_14/TensorArrayV2_1/element_shape?
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/time?
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter?
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_14_while_body_90278*$
condR
lstm_14_while_cond_90277*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_14/while?
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack?
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_14/strided_slice_3/stack?
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1?
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2?
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_14/strided_slice_3?
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm?
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtime?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul lstm_14/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_13/Relu?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_14/Softmax?
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
while_cond_92305
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92305___redundant_placeholder03
/while_while_cond_92305___redundant_placeholder13
/while_while_cond_92305___redundant_placeholder23
/while_while_cond_92305___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
}
(__inference_dense_13_layer_call_fn_92934

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_896322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89716
lstm_12_input
lstm_12_89679
lstm_12_89681
lstm_12_89683
lstm_13_89686
lstm_13_89688
lstm_13_89690
lstm_14_89693
lstm_14_89695
lstm_14_89697
dense_12_89700
dense_12_89702
dense_13_89705
dense_13_89707
dense_14_89710
dense_14_89712
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_89679lstm_12_89681lstm_12_89683*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_888942!
lstm_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0lstm_13_89686lstm_13_89688lstm_13_89690*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_892292!
lstm_13/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0lstm_14_89693lstm_14_89695lstm_14_89697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_895642!
lstm_14/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_12_89700dense_12_89702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_896052"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_89705dense_13_89707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_896322"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_89710dense_14_89712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_896592"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
?
?
while_cond_88506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_88506___redundant_placeholder03
/while_while_cond_88506___redundant_placeholder13
/while_while_cond_88506___redundant_placeholder23
/while_while_cond_88506___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_88894

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_88809*
condR
while_cond_88808*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
? 
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89759

inputs
lstm_12_89722
lstm_12_89724
lstm_12_89726
lstm_13_89729
lstm_13_89731
lstm_13_89733
lstm_14_89736
lstm_14_89738
lstm_14_89740
dense_12_89743
dense_12_89745
dense_13_89748
dense_13_89750
dense_14_89753
dense_14_89755
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_89722lstm_12_89724lstm_12_89726*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_887412!
lstm_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0lstm_13_89729lstm_13_89731lstm_13_89733*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_890762!
lstm_13/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0lstm_14_89736lstm_14_89738lstm_14_89740*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_894112!
lstm_14/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_12_89743dense_12_89745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_896052"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_89748dense_13_89750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_896322"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_89753dense_14_89755*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_896592"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
while_cond_91649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91649___redundant_placeholder03
/while_while_cond_91649___redundant_placeholder13
/while_while_cond_91649___redundant_placeholder23
/while_while_cond_91649___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_12_layer_call_fn_91254
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_873562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????~
"
_user_specified_name
inputs/0
?

?
lstm_14_while_cond_90277,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1C
?lstm_14_while_lstm_14_while_cond_90277___redundant_placeholder0C
?lstm_14_while_lstm_14_while_cond_90277___redundant_placeholder1C
?lstm_14_while_lstm_14_while_cond_90277___redundant_placeholder2C
?lstm_14_while_lstm_14_while_cond_90277___redundant_placeholder3
lstm_14_while_identity
?
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_92459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
%sequential_4_lstm_12_while_cond_86350F
Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counterL
Hsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations*
&sequential_4_lstm_12_while_placeholder,
(sequential_4_lstm_12_while_placeholder_1,
(sequential_4_lstm_12_while_placeholder_2,
(sequential_4_lstm_12_while_placeholder_3H
Dsequential_4_lstm_12_while_less_sequential_4_lstm_12_strided_slice_1]
Ysequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_86350___redundant_placeholder0]
Ysequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_86350___redundant_placeholder1]
Ysequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_86350___redundant_placeholder2]
Ysequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_86350___redundant_placeholder3'
#sequential_4_lstm_12_while_identity
?
sequential_4/lstm_12/while/LessLess&sequential_4_lstm_12_while_placeholderDsequential_4_lstm_12_while_less_sequential_4_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_4/lstm_12/while/Less?
#sequential_4/lstm_12/while/IdentityIdentity#sequential_4/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_4/lstm_12/while/Identity"S
#sequential_4_lstm_12_while_identity,sequential_4/lstm_12/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_91978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
%sequential_4_lstm_13_while_cond_86499F
Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counterL
Hsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations*
&sequential_4_lstm_13_while_placeholder,
(sequential_4_lstm_13_while_placeholder_1,
(sequential_4_lstm_13_while_placeholder_2,
(sequential_4_lstm_13_while_placeholder_3H
Dsequential_4_lstm_13_while_less_sequential_4_lstm_13_strided_slice_1]
Ysequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_86499___redundant_placeholder0]
Ysequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_86499___redundant_placeholder1]
Ysequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_86499___redundant_placeholder2]
Ysequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_86499___redundant_placeholder3'
#sequential_4_lstm_13_while_identity
?
sequential_4/lstm_13/while/LessLess&sequential_4_lstm_13_while_placeholderDsequential_4_lstm_13_while_less_sequential_4_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_4/lstm_13/while/Less?
#sequential_4/lstm_13/while/IdentityIdentity#sequential_4/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_4/lstm_13/while/Identity"S
#sequential_4_lstm_13_while_identity,sequential_4/lstm_13/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_12_layer_call_fn_91582

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_888942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
'__inference_lstm_14_layer_call_fn_92883
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_884442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?D
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_87966

inputs
lstm_cell_13_87884
lstm_cell_13_87886
lstm_cell_13_87888
identity??$lstm_cell_13/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_87884lstm_cell_13_87886lstm_cell_13_87888*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874712&
$lstm_cell_13/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_87884lstm_cell_13_87886lstm_cell_13_87888*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_87897*
condR
while_cond_87896*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0%^lstm_cell_13/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?D
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_88444

inputs
lstm_cell_14_88362
lstm_cell_14_88364
lstm_cell_14_88366
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_88362lstm_cell_14_88364lstm_cell_14_88366*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880482&
$lstm_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_88362lstm_cell_14_88364lstm_cell_14_88366*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_88375*
condR
while_cond_88374*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_89605

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_lstm_14_layer_call_fn_92566

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_895642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
while_cond_92130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92130___redundant_placeholder03
/while_while_cond_92130___redundant_placeholder13
/while_while_cond_92130___redundant_placeholder23
/while_while_cond_92130___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_92634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_92905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?%
?
while_body_87287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_12_87311_0
while_lstm_cell_12_87313_0
while_lstm_cell_12_87315_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_12_87311
while_lstm_cell_12_87313
while_lstm_cell_12_87315??*while/lstm_cell_12/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_87311_0while_lstm_cell_12_87313_0while_lstm_cell_12_87315_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868612,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_12_87311while_lstm_cell_12_87311_0"6
while_lstm_cell_12_87313while_lstm_cell_12_87313_0"6
while_lstm_cell_12_87315while_lstm_cell_12_87315_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_88374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_88374___redundant_placeholder03
/while_while_cond_88374___redundant_placeholder13
/while_while_cond_88374___redundant_placeholder23
/while_while_cond_88374___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?b
?
%sequential_4_lstm_14_while_body_86649F
Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counterL
Hsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations*
&sequential_4_lstm_14_while_placeholder,
(sequential_4_lstm_14_while_placeholder_1,
(sequential_4_lstm_14_while_placeholder_2,
(sequential_4_lstm_14_while_placeholder_3E
Asequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1_0?
}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0L
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0N
Jsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0M
Isequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0'
#sequential_4_lstm_14_while_identity)
%sequential_4_lstm_14_while_identity_1)
%sequential_4_lstm_14_while_identity_2)
%sequential_4_lstm_14_while_identity_3)
%sequential_4_lstm_14_while_identity_4)
%sequential_4_lstm_14_while_identity_5C
?sequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1
{sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensorJ
Fsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceL
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceK
Gsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp??sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
Lsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2N
Lsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_14_while_placeholderUsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem?
=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
.sequential_4/lstm_14/while/lstm_cell_14/MatMulMatMulEsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_4/lstm_14/while/lstm_cell_14/MatMul?
?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02A
?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
0sequential_4/lstm_14/while/lstm_cell_14/MatMul_1MatMul(sequential_4_lstm_14_while_placeholder_2Gsequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_4/lstm_14/while/lstm_cell_14/MatMul_1?
+sequential_4/lstm_14/while/lstm_cell_14/addAddV28sequential_4/lstm_14/while/lstm_cell_14/MatMul:product:0:sequential_4/lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_14/while/lstm_cell_14/add?
>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
/sequential_4/lstm_14/while/lstm_cell_14/BiasAddBiasAdd/sequential_4/lstm_14/while/lstm_cell_14/add:z:0Fsequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_4/lstm_14/while/lstm_cell_14/BiasAdd?
-sequential_4/lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_4/lstm_14/while/lstm_cell_14/Const?
7sequential_4/lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_4/lstm_14/while/lstm_cell_14/split/split_dim?
-sequential_4/lstm_14/while/lstm_cell_14/splitSplit@sequential_4/lstm_14/while/lstm_cell_14/split/split_dim:output:08sequential_4/lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2/
-sequential_4/lstm_14/while/lstm_cell_14/split?
/sequential_4/lstm_14/while/lstm_cell_14/SigmoidSigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@21
/sequential_4/lstm_14/while/lstm_cell_14/Sigmoid?
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@23
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_1?
+sequential_4/lstm_14/while/lstm_cell_14/mulMul5sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_1:y:0(sequential_4_lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_14/while/lstm_cell_14/mul?
,sequential_4/lstm_14/while/lstm_cell_14/ReluRelu6sequential_4/lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2.
,sequential_4/lstm_14/while/lstm_cell_14/Relu?
-sequential_4/lstm_14/while/lstm_cell_14/mul_1Mul3sequential_4/lstm_14/while/lstm_cell_14/Sigmoid:y:0:sequential_4/lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_14/while/lstm_cell_14/mul_1?
-sequential_4/lstm_14/while/lstm_cell_14/add_1AddV2/sequential_4/lstm_14/while/lstm_cell_14/mul:z:01sequential_4/lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_14/while/lstm_cell_14/add_1?
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@23
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_2?
.sequential_4/lstm_14/while/lstm_cell_14/Relu_1Relu1sequential_4/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@20
.sequential_4/lstm_14/while/lstm_cell_14/Relu_1?
-sequential_4/lstm_14/while/lstm_cell_14/mul_2Mul5sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_2:y:0<sequential_4/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_14/while/lstm_cell_14/mul_2?
?sequential_4/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_14_while_placeholder_1&sequential_4_lstm_14_while_placeholder1sequential_4/lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_4/lstm_14/while/TensorArrayV2Write/TensorListSetItem?
 sequential_4/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_4/lstm_14/while/add/y?
sequential_4/lstm_14/while/addAddV2&sequential_4_lstm_14_while_placeholder)sequential_4/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_14/while/add?
"sequential_4/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_4/lstm_14/while/add_1/y?
 sequential_4/lstm_14/while/add_1AddV2Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counter+sequential_4/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_14/while/add_1?
#sequential_4/lstm_14/while/IdentityIdentity$sequential_4/lstm_14/while/add_1:z:0?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_4/lstm_14/while/Identity?
%sequential_4/lstm_14/while/Identity_1IdentityHsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_14/while/Identity_1?
%sequential_4/lstm_14/while/Identity_2Identity"sequential_4/lstm_14/while/add:z:0?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_14/while/Identity_2?
%sequential_4/lstm_14/while/Identity_3IdentityOsequential_4/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_14/while/Identity_3?
%sequential_4/lstm_14/while/Identity_4Identity1sequential_4/lstm_14/while/lstm_cell_14/mul_2:z:0?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_14/while/Identity_4?
%sequential_4/lstm_14/while/Identity_5Identity1sequential_4/lstm_14/while/lstm_cell_14/add_1:z:0?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_14/while/Identity_5"S
#sequential_4_lstm_14_while_identity,sequential_4/lstm_14/while/Identity:output:0"W
%sequential_4_lstm_14_while_identity_1.sequential_4/lstm_14/while/Identity_1:output:0"W
%sequential_4_lstm_14_while_identity_2.sequential_4/lstm_14/while/Identity_2:output:0"W
%sequential_4_lstm_14_while_identity_3.sequential_4/lstm_14/while/Identity_3:output:0"W
%sequential_4_lstm_14_while_identity_4.sequential_4/lstm_14/while/Identity_4:output:0"W
%sequential_4_lstm_14_while_identity_5.sequential_4/lstm_14/while/Identity_5:output:0"?
Gsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resourceIsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"?
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceJsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"?
Fsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceHsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
?sequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1Asequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1_0"?
{sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2?
>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2?
?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_89478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_89478___redundant_placeholder03
/while_while_cond_89478___redundant_placeholder13
/while_while_cond_89478___redundant_placeholder23
/while_while_cond_89478___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_13_layer_call_fn_92227

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_890762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?%
?
while_body_87765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_13_87789_0
while_lstm_cell_13_87791_0
while_lstm_cell_13_87793_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_13_87789
while_lstm_cell_13_87791
while_lstm_cell_13_87793??*while/lstm_cell_13/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_87789_0while_lstm_cell_13_87791_0while_lstm_cell_13_87793_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874382,
*while/lstm_cell_13/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1+^while/lstm_cell_13/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2+^while/lstm_cell_13/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_13_87789while_lstm_cell_13_87789_0"6
while_lstm_cell_13_87791while_lstm_cell_13_87791_0"6
while_lstm_cell_13_87793while_lstm_cell_13_87793_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91232
inputs_0/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91147*
condR
while_cond_91146*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????~
"
_user_specified_name
inputs/0
?	
?
C__inference_dense_14_layer_call_and_return_conditional_losses_92945

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_92987

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?D
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_87356

inputs
lstm_cell_12_87274
lstm_cell_12_87276
lstm_cell_12_87278
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_87274lstm_cell_12_87276lstm_cell_12_87278*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868612&
$lstm_cell_12/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_87274lstm_cell_12_87276lstm_cell_12_87278*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_87287*
condR
while_cond_87286*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0%^lstm_cell_12/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????~
 
_user_specified_nameinputs
? 
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89834

inputs
lstm_12_89797
lstm_12_89799
lstm_12_89801
lstm_13_89804
lstm_13_89806
lstm_13_89808
lstm_14_89811
lstm_14_89813
lstm_14_89815
dense_12_89818
dense_12_89820
dense_13_89823
dense_13_89825
dense_14_89828
dense_14_89830
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_89797lstm_12_89799lstm_12_89801*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_888942!
lstm_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0lstm_13_89804lstm_13_89806lstm_13_89808*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_892292!
lstm_13/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0lstm_14_89811lstm_14_89813lstm_14_89815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_14_layer_call_and_return_conditional_losses_895642!
lstm_14/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_12_89818dense_12_89820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_896052"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_89823dense_13_89825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_896322"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_89828dense_14_89830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_896592"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?

?
lstm_12_while_cond_89979,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1C
?lstm_12_while_lstm_12_while_cond_89979___redundant_placeholder0C
?lstm_12_while_lstm_12_while_cond_89979___redundant_placeholder1C
?lstm_12_while_lstm_12_while_cond_89979___redundant_placeholder2C
?lstm_12_while_lstm_12_while_cond_89979___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91407

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91322*
condR
while_cond_91321*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?b
?
%sequential_4_lstm_13_while_body_86500F
Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counterL
Hsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations*
&sequential_4_lstm_13_while_placeholder,
(sequential_4_lstm_13_while_placeholder_1,
(sequential_4_lstm_13_while_placeholder_2,
(sequential_4_lstm_13_while_placeholder_3E
Asequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1_0?
}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0L
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0N
Jsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0M
Isequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0'
#sequential_4_lstm_13_while_identity)
%sequential_4_lstm_13_while_identity_1)
%sequential_4_lstm_13_while_identity_2)
%sequential_4_lstm_13_while_identity_3)
%sequential_4_lstm_13_while_identity_4)
%sequential_4_lstm_13_while_identity_5C
?sequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1
{sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensorJ
Fsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceL
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceK
Gsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource??>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp??sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
Lsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2N
Lsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_13_while_placeholderUsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02@
>sequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem?
=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02?
=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
.sequential_4/lstm_13/while/lstm_cell_13/MatMulMatMulEsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_4/lstm_13/while/lstm_cell_13/MatMul?
?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02A
?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
0sequential_4/lstm_13/while/lstm_cell_13/MatMul_1MatMul(sequential_4_lstm_13_while_placeholder_2Gsequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_4/lstm_13/while/lstm_cell_13/MatMul_1?
+sequential_4/lstm_13/while/lstm_cell_13/addAddV28sequential_4/lstm_13/while/lstm_cell_13/MatMul:product:0:sequential_4/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_13/while/lstm_cell_13/add?
>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
/sequential_4/lstm_13/while/lstm_cell_13/BiasAddBiasAdd/sequential_4/lstm_13/while/lstm_cell_13/add:z:0Fsequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_4/lstm_13/while/lstm_cell_13/BiasAdd?
-sequential_4/lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_4/lstm_13/while/lstm_cell_13/Const?
7sequential_4/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_4/lstm_13/while/lstm_cell_13/split/split_dim?
-sequential_4/lstm_13/while/lstm_cell_13/splitSplit@sequential_4/lstm_13/while/lstm_cell_13/split/split_dim:output:08sequential_4/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2/
-sequential_4/lstm_13/while/lstm_cell_13/split?
/sequential_4/lstm_13/while/lstm_cell_13/SigmoidSigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????21
/sequential_4/lstm_13/while/lstm_cell_13/Sigmoid?
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????23
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_1?
+sequential_4/lstm_13/while/lstm_cell_13/mulMul5sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0(sequential_4_lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_13/while/lstm_cell_13/mul?
,sequential_4/lstm_13/while/lstm_cell_13/ReluRelu6sequential_4/lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2.
,sequential_4/lstm_13/while/lstm_cell_13/Relu?
-sequential_4/lstm_13/while/lstm_cell_13/mul_1Mul3sequential_4/lstm_13/while/lstm_cell_13/Sigmoid:y:0:sequential_4/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_4/lstm_13/while/lstm_cell_13/mul_1?
-sequential_4/lstm_13/while/lstm_cell_13/add_1AddV2/sequential_4/lstm_13/while/lstm_cell_13/mul:z:01sequential_4/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_4/lstm_13/while/lstm_cell_13/add_1?
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????23
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_2?
.sequential_4/lstm_13/while/lstm_cell_13/Relu_1Relu1sequential_4/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_4/lstm_13/while/lstm_cell_13/Relu_1?
-sequential_4/lstm_13/while/lstm_cell_13/mul_2Mul5sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_2:y:0<sequential_4/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_4/lstm_13/while/lstm_cell_13/mul_2?
?sequential_4/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_13_while_placeholder_1&sequential_4_lstm_13_while_placeholder1sequential_4/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_4/lstm_13/while/TensorArrayV2Write/TensorListSetItem?
 sequential_4/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_4/lstm_13/while/add/y?
sequential_4/lstm_13/while/addAddV2&sequential_4_lstm_13_while_placeholder)sequential_4/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_13/while/add?
"sequential_4/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_4/lstm_13/while/add_1/y?
 sequential_4/lstm_13/while/add_1AddV2Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counter+sequential_4/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_13/while/add_1?
#sequential_4/lstm_13/while/IdentityIdentity$sequential_4/lstm_13/while/add_1:z:0?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_4/lstm_13/while/Identity?
%sequential_4/lstm_13/while/Identity_1IdentityHsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_13/while/Identity_1?
%sequential_4/lstm_13/while/Identity_2Identity"sequential_4/lstm_13/while/add:z:0?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_13/while/Identity_2?
%sequential_4/lstm_13/while/Identity_3IdentityOsequential_4/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_13/while/Identity_3?
%sequential_4/lstm_13/while/Identity_4Identity1sequential_4/lstm_13/while/lstm_cell_13/mul_2:z:0?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_13/while/Identity_4?
%sequential_4/lstm_13/while/Identity_5Identity1sequential_4/lstm_13/while/lstm_cell_13/add_1:z:0?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2'
%sequential_4/lstm_13/while/Identity_5"S
#sequential_4_lstm_13_while_identity,sequential_4/lstm_13/while/Identity:output:0"W
%sequential_4_lstm_13_while_identity_1.sequential_4/lstm_13/while/Identity_1:output:0"W
%sequential_4_lstm_13_while_identity_2.sequential_4/lstm_13/while/Identity_2:output:0"W
%sequential_4_lstm_13_while_identity_3.sequential_4/lstm_13/while/Identity_3:output:0"W
%sequential_4_lstm_13_while_identity_4.sequential_4/lstm_13/while/Identity_4:output:0"W
%sequential_4_lstm_13_while_identity_5.sequential_4/lstm_13/while/Identity_5:output:0"?
Gsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceIsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"?
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceJsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"?
Fsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceHsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
?sequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1Asequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1_0"?
{sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2?
>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2?
?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?O
?	
lstm_12_while_body_89980,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0A
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0@
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor=
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource?
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource>
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
 lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_12/while/lstm_cell_12/Const?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1?
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1?
lstm_12/while/Identity_2Identitylstm_12/while/add:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:02^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_5"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92872
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_92787*
condR
while_cond_92786*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93120

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_92633
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92633___redundant_placeholder03
/while_while_cond_92633___redundant_placeholder13
/while_while_cond_92633___redundant_placeholder23
/while_while_cond_92633___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?%
?
while_body_87897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_13_87921_0
while_lstm_cell_13_87923_0
while_lstm_cell_13_87925_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_13_87921
while_lstm_cell_13_87923
while_lstm_cell_13_87925??*while/lstm_cell_13/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_87921_0while_lstm_cell_13_87923_0while_lstm_cell_13_87925_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874712,
*while/lstm_cell_13/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1+^while/lstm_cell_13/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2+^while/lstm_cell_13/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_13_87921while_lstm_cell_13_87921_0"6
while_lstm_cell_13_87923while_lstm_cell_13_87923_0"6
while_lstm_cell_13_87925while_lstm_cell_13_87925_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_88048

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_92063

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91978*
condR
while_cond_91977*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_88741

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_88656*
condR
while_cond_88655*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_86828

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?C
?
while_body_92131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_89076

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_88991*
condR
while_cond_88990*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?C
?
while_body_91475
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?D
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_87224

inputs
lstm_cell_12_87142
lstm_cell_12_87144
lstm_cell_12_87146
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_87142lstm_cell_12_87144lstm_cell_12_87146*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868282&
$lstm_cell_12/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_87142lstm_cell_12_87144lstm_cell_12_87146*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_87155*
condR
while_cond_87154*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0%^lstm_cell_12/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????~
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_86861

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????~:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?C
?
while_body_91322
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?b
?
%sequential_4_lstm_12_while_body_86351F
Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counterL
Hsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations*
&sequential_4_lstm_12_while_placeholder,
(sequential_4_lstm_12_while_placeholder_1,
(sequential_4_lstm_12_while_placeholder_2,
(sequential_4_lstm_12_while_placeholder_3E
Asequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1_0?
}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0L
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0N
Jsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0M
Isequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0'
#sequential_4_lstm_12_while_identity)
%sequential_4_lstm_12_while_identity_1)
%sequential_4_lstm_12_while_identity_2)
%sequential_4_lstm_12_while_identity_3)
%sequential_4_lstm_12_while_identity_4)
%sequential_4_lstm_12_while_identity_5C
?sequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1
{sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensorJ
Fsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceL
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceK
Gsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp??sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
Lsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   2N
Lsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_12_while_placeholderUsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02@
>sequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem?
=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02?
=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
.sequential_4/lstm_12/while/lstm_cell_12/MatMulMatMulEsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_4/lstm_12/while/lstm_cell_12/MatMul?
?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02A
?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
0sequential_4/lstm_12/while/lstm_cell_12/MatMul_1MatMul(sequential_4_lstm_12_while_placeholder_2Gsequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_4/lstm_12/while/lstm_cell_12/MatMul_1?
+sequential_4/lstm_12/while/lstm_cell_12/addAddV28sequential_4/lstm_12/while/lstm_cell_12/MatMul:product:0:sequential_4/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_4/lstm_12/while/lstm_cell_12/add?
>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
/sequential_4/lstm_12/while/lstm_cell_12/BiasAddBiasAdd/sequential_4/lstm_12/while/lstm_cell_12/add:z:0Fsequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_4/lstm_12/while/lstm_cell_12/BiasAdd?
-sequential_4/lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_4/lstm_12/while/lstm_cell_12/Const?
7sequential_4/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_4/lstm_12/while/lstm_cell_12/split/split_dim?
-sequential_4/lstm_12/while/lstm_cell_12/splitSplit@sequential_4/lstm_12/while/lstm_cell_12/split/split_dim:output:08sequential_4/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2/
-sequential_4/lstm_12/while/lstm_cell_12/split?
/sequential_4/lstm_12/while/lstm_cell_12/SigmoidSigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@21
/sequential_4/lstm_12/while/lstm_cell_12/Sigmoid?
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@23
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_1?
+sequential_4/lstm_12/while/lstm_cell_12/mulMul5sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0(sequential_4_lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2-
+sequential_4/lstm_12/while/lstm_cell_12/mul?
,sequential_4/lstm_12/while/lstm_cell_12/ReluRelu6sequential_4/lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2.
,sequential_4/lstm_12/while/lstm_cell_12/Relu?
-sequential_4/lstm_12/while/lstm_cell_12/mul_1Mul3sequential_4/lstm_12/while/lstm_cell_12/Sigmoid:y:0:sequential_4/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_12/while/lstm_cell_12/mul_1?
-sequential_4/lstm_12/while/lstm_cell_12/add_1AddV2/sequential_4/lstm_12/while/lstm_cell_12/mul:z:01sequential_4/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_12/while/lstm_cell_12/add_1?
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@23
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_2?
.sequential_4/lstm_12/while/lstm_cell_12/Relu_1Relu1sequential_4/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@20
.sequential_4/lstm_12/while/lstm_cell_12/Relu_1?
-sequential_4/lstm_12/while/lstm_cell_12/mul_2Mul5sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0<sequential_4/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2/
-sequential_4/lstm_12/while/lstm_cell_12/mul_2?
?sequential_4/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_12_while_placeholder_1&sequential_4_lstm_12_while_placeholder1sequential_4/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_4/lstm_12/while/TensorArrayV2Write/TensorListSetItem?
 sequential_4/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_4/lstm_12/while/add/y?
sequential_4/lstm_12/while/addAddV2&sequential_4_lstm_12_while_placeholder)sequential_4/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_12/while/add?
"sequential_4/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_4/lstm_12/while/add_1/y?
 sequential_4/lstm_12/while/add_1AddV2Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counter+sequential_4/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_12/while/add_1?
#sequential_4/lstm_12/while/IdentityIdentity$sequential_4/lstm_12/while/add_1:z:0?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_4/lstm_12/while/Identity?
%sequential_4/lstm_12/while/Identity_1IdentityHsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_12/while/Identity_1?
%sequential_4/lstm_12/while/Identity_2Identity"sequential_4/lstm_12/while/add:z:0?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_12/while/Identity_2?
%sequential_4/lstm_12/while/Identity_3IdentityOsequential_4/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_4/lstm_12/while/Identity_3?
%sequential_4/lstm_12/while/Identity_4Identity1sequential_4/lstm_12/while/lstm_cell_12/mul_2:z:0?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_12/while/Identity_4?
%sequential_4/lstm_12/while/Identity_5Identity1sequential_4/lstm_12/while/lstm_cell_12/add_1:z:0?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2'
%sequential_4/lstm_12/while/Identity_5"S
#sequential_4_lstm_12_while_identity,sequential_4/lstm_12/while/Identity:output:0"W
%sequential_4_lstm_12_while_identity_1.sequential_4/lstm_12/while/Identity_1:output:0"W
%sequential_4_lstm_12_while_identity_2.sequential_4/lstm_12/while/Identity_2:output:0"W
%sequential_4_lstm_12_while_identity_3.sequential_4/lstm_12/while/Identity_3:output:0"W
%sequential_4_lstm_12_while_identity_4.sequential_4/lstm_12/while/Identity_4:output:0"W
%sequential_4_lstm_12_while_identity_5.sequential_4/lstm_12/while/Identity_5:output:0"?
Gsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceIsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"?
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceJsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"?
Fsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceHsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
?sequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1Asequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1_0"?
{sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2?
>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2?
?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?D
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_88576

inputs
lstm_cell_14_88494
lstm_cell_14_88496
lstm_cell_14_88498
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_88494lstm_cell_14_88496lstm_cell_14_88498*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880812&
$lstm_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_88494lstm_cell_14_88496lstm_cell_14_88498*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_88507*
condR
while_cond_88506*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?C
?
while_body_88991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
#__inference_signature_wrapper_89912
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_867552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
~
'
_user_specified_namelstm_12_input
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_92925

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?[
?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91079
inputs_0/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????~2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????~*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	~?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_90994*
condR
while_cond_90993*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????~
"
_user_specified_name
inputs/0
?
?
while_cond_87154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_87154___redundant_placeholder03
/while_while_cond_87154___redundant_placeholder13
/while_while_cond_87154___redundant_placeholder23
/while_while_cond_87154___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_14_layer_call_fn_93237

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?C
?
while_body_92787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_13_layer_call_fn_93137

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_89325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_89325___redundant_placeholder03
/while_while_cond_89325___redundant_placeholder13
/while_while_cond_89325___redundant_placeholder23
/while_while_cond_89325___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?

?
lstm_13_while_cond_90128,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1C
?lstm_13_while_lstm_13_while_cond_90128___redundant_placeholder0C
?lstm_13_while_lstm_13_while_cond_90128___redundant_placeholder1C
?lstm_13_while_lstm_13_while_cond_90128___redundant_placeholder2C
?lstm_13_while_lstm_13_while_cond_90128___redundant_placeholder3
lstm_13_while_identity
?
lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_89326
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
%sequential_4_lstm_14_while_cond_86648F
Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counterL
Hsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations*
&sequential_4_lstm_14_while_placeholder,
(sequential_4_lstm_14_while_placeholder_1,
(sequential_4_lstm_14_while_placeholder_2,
(sequential_4_lstm_14_while_placeholder_3H
Dsequential_4_lstm_14_while_less_sequential_4_lstm_14_strided_slice_1]
Ysequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_86648___redundant_placeholder0]
Ysequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_86648___redundant_placeholder1]
Ysequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_86648___redundant_placeholder2]
Ysequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_86648___redundant_placeholder3'
#sequential_4_lstm_14_while_identity
?
sequential_4/lstm_14/while/LessLess&sequential_4_lstm_14_while_placeholderDsequential_4_lstm_14_while_less_sequential_4_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_4/lstm_14/while/Less?
#sequential_4/lstm_14/while/IdentityIdentity#sequential_4/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_4/lstm_14/while/Identity"S
#sequential_4_lstm_14_while_identity,sequential_4/lstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_87438

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?O
?	
lstm_13_while_body_90601,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0A
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0@
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor=
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource?
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource>
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource??1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItem?
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/MatMul?
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_13/while/lstm_cell_13/MatMul_1?
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/add?
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/BiasAdd?
 lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_13/while/lstm_cell_13/Const?
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim?
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_13/while/lstm_cell_13/split?
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/Sigmoid?
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_1?
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/mul?
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_13/while/lstm_cell_13/Relu?
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_1?
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/add_1?
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_2?
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/Relu_1?
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_2?
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y?
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y?
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1?
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity?
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1?
lstm_13/while/Identity_2Identitylstm_13/while/add:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2?
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3?
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_4?
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_5"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_91888
inputs_0/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_91803*
condR
while_cond_91802*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
'__inference_lstm_12_layer_call_fn_91571

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_887412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
~:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93187

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?%
?
while_body_88375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_14_88399_0
while_lstm_cell_14_88401_0
while_lstm_cell_14_88403_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_14_88399
while_lstm_cell_14_88401
while_lstm_cell_14_88403??*while/lstm_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_88399_0while_lstm_cell_14_88401_0while_lstm_cell_14_88403_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880482,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_14_88399while_lstm_cell_14_88399_0"6
while_lstm_cell_14_88401while_lstm_cell_14_88401_0"6
while_lstm_cell_14_88403while_lstm_cell_14_88403_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93220

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?
?
'__inference_lstm_12_layer_call_fn_91243
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_12_layer_call_and_return_conditional_losses_872242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????~:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????~
"
_user_specified_name
inputs/0
?C
?
while_body_89479
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_88655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_88655___redundant_placeholder03
/while_while_cond_88655___redundant_placeholder13
/while_while_cond_88655___redundant_placeholder23
/while_while_cond_88655___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_87471

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?C
?
while_body_91650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_89411

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:
??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_89326*
condR
while_cond_89325*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????
?:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?C
?
while_body_88809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?[
?
B__inference_lstm_13_layer_call_and_return_conditional_losses_92216

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
?????????@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_92131*
condR
while_cond_92130*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
@:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93087

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_88808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_88808___redundant_placeholder03
/while_while_cond_88808___redundant_placeholder13
/while_while_cond_88808___redundant_placeholder23
/while_while_cond_88808___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_91321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91321___redundant_placeholder03
/while_while_cond_91321___redundant_placeholder13
/while_while_cond_91321___redundant_placeholder23
/while_while_cond_91321___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_13_layer_call_fn_91910
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_13_layer_call_and_return_conditional_losses_879662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?C
?
while_body_92306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_88990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_88990___redundant_placeholder03
/while_while_cond_88990___redundant_placeholder13
/while_while_cond_88990___redundant_placeholder23
/while_while_cond_88990___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
,__inference_sequential_4_layer_call_fn_90926

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_898342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?
?
while_cond_87764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_87764___redundant_placeholder03
/while_while_cond_87764___redundant_placeholder13
/while_while_cond_87764___redundant_placeholder23
/while_while_cond_87764___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_87896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_87896___redundant_placeholder03
/while_while_cond_87896___redundant_placeholder13
/while_while_cond_87896___redundant_placeholder23
/while_while_cond_87896___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?[
?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92719
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_92634*
condR
while_cond_92633*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?%
?
while_body_87155
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_12_87179_0
while_lstm_cell_12_87181_0
while_lstm_cell_12_87183_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_12_87179
while_lstm_cell_12_87181
while_lstm_cell_12_87183??*while/lstm_cell_12/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_87179_0while_lstm_cell_12_87181_0while_lstm_cell_12_87183_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_868282,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_12_87179while_lstm_cell_12_87179_0"6
while_lstm_cell_12_87181while_lstm_cell_12_87181_0"6
while_lstm_cell_12_87183while_lstm_cell_12_87183_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?o
?
__inference__traced_save_93439
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableopD
@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop8
4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop:
6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop:
6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableopD
@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop8
4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@@:@:@ : : :: : : : : :	~?:	@?:?:	@?:
??:?:
??:	@?:?: : : : :@@:@:@ : : ::	~?:	@?:?:	@?:
??:?:
??:	@?:?:@@:@:@ : : ::	~?:	@?:?:	@?:
??:?:
??:	@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	~?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	@?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	@?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	~?:% !

_output_shapes
:	@?:!!

_output_shapes	
:?:%"!

_output_shapes
:	@?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:%&!

_output_shapes
:	@?:!'

_output_shapes	
:?:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@ : +

_output_shapes
: :$, 

_output_shapes

: : -

_output_shapes
::%.!

_output_shapes
:	~?:%/!

_output_shapes
:	@?:!0

_output_shapes	
:?:%1!

_output_shapes
:	@?:&2"
 
_output_shapes
:
??:!3

_output_shapes	
:?:&4"
 
_output_shapes
:
??:%5!

_output_shapes
:	@?:!6

_output_shapes	
:?:7

_output_shapes
: 
?C
?
while_body_90994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????~   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????~*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	~?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_91977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91977___redundant_placeholder03
/while_while_cond_91977___redundant_placeholder13
/while_while_cond_91977___redundant_placeholder23
/while_while_cond_91977___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_13_layer_call_fn_93154

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_874712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????@:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_92458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92458___redundant_placeholder03
/while_while_cond_92458___redundant_placeholder13
/while_while_cond_92458___redundant_placeholder23
/while_while_cond_92458___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?

?
,__inference_sequential_4_layer_call_fn_90891

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_897592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????
~:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
~
 
_user_specified_nameinputs
?C
?
while_body_89144
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource??)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_87286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_87286___redundant_placeholder03
/while_while_cond_87286___redundant_placeholder13
/while_while_cond_87286___redundant_placeholder23
/while_while_cond_87286___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_88081

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????@:?????????@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?%
?
while_body_88507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_14_88531_0
while_lstm_cell_14_88533_0
while_lstm_cell_14_88535_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_14_88531
while_lstm_cell_14_88533
while_lstm_cell_14_88535??*while/lstm_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_88531_0while_lstm_cell_14_88533_0while_lstm_cell_14_88535_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_880812,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_14_88531while_lstm_cell_14_88531_0"6
while_lstm_cell_14_88533while_lstm_cell_14_88533_0"6
while_lstm_cell_14_88535while_lstm_cell_14_88535_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_12_input:
serving_default_lstm_12_input:0?????????
~<
dense_140
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?K
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?H
_tf_keras_sequential?H{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 126]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 126]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 126]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 126]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 126]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 128]}}
?

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
1iter

2beta_1

3beta_2
	4decay
5learning_ratem? m?%m?&m?+m?,m?6m?7m?8m?9m?:m?;m?<m?=m?>m?v? v?%v?&v?+v?,v?6v?7v?8v?9v?:v?;v?<v?=v?>v?"
	optimizer
 "
trackable_list_wrapper
?
60
71
82
93
:4
;5
<6
=7
>8
9
 10
%11
&12
+13
,14"
trackable_list_wrapper
?
60
71
82
93
:4
;5
<6
=7
>8
9
 10
%11
&12
+13
,14"
trackable_list_wrapper
?
regularization_losses
?non_trainable_variables
@metrics
		variables
Alayer_metrics

trainable_variables

Blayers
Clayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

6kernel
7recurrent_kernel
8bias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
?
regularization_losses
Hnon_trainable_variables

Istates
Jmetrics
	variables
Klayer_metrics
trainable_variables

Llayers
Mlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

9kernel
:recurrent_kernel
;bias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
?
regularization_losses
Rnon_trainable_variables

Sstates
Tmetrics
	variables
Ulayer_metrics
trainable_variables

Vlayers
Wlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

<kernel
=recurrent_kernel
>bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
?
regularization_losses
\non_trainable_variables

]states
^metrics
	variables
_layer_metrics
trainable_variables

`layers
alayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_12/kernel
:@2dense_12/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
!regularization_losses
bnon_trainable_variables
cmetrics
"	variables
dlayer_metrics
#trainable_variables

elayers
flayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@ 2dense_13/kernel
: 2dense_13/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'regularization_losses
gnon_trainable_variables
hmetrics
(	variables
ilayer_metrics
)trainable_variables

jlayers
klayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
-regularization_losses
lnon_trainable_variables
mmetrics
.	variables
nlayer_metrics
/trainable_variables

olayers
player_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	~?2lstm_12/lstm_cell_12/kernel
8:6	@?2%lstm_12/lstm_cell_12/recurrent_kernel
(:&?2lstm_12/lstm_cell_12/bias
.:,	@?2lstm_13/lstm_cell_13/kernel
9:7
??2%lstm_13/lstm_cell_13/recurrent_kernel
(:&?2lstm_13/lstm_cell_13/bias
/:-
??2lstm_14/lstm_cell_14/kernel
8:6	@?2%lstm_14/lstm_cell_14/recurrent_kernel
(:&?2lstm_14/lstm_cell_14/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
?
Dregularization_losses
snon_trainable_variables
tmetrics
E	variables
ulayer_metrics
Ftrainable_variables

vlayers
wlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
?
Nregularization_losses
xnon_trainable_variables
ymetrics
O	variables
zlayer_metrics
Ptrainable_variables

{layers
|layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
?
Xregularization_losses
}non_trainable_variables
~metrics
Y	variables
layer_metrics
Ztrainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$@@2Adam/dense_12/kernel/m
 :@2Adam/dense_12/bias/m
&:$@ 2Adam/dense_13/kernel/m
 : 2Adam/dense_13/bias/m
&:$ 2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
3:1	~?2"Adam/lstm_12/lstm_cell_12/kernel/m
=:;	@?2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+?2 Adam/lstm_12/lstm_cell_12/bias/m
3:1	@?2"Adam/lstm_13/lstm_cell_13/kernel/m
>:<
??2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+?2 Adam/lstm_13/lstm_cell_13/bias/m
4:2
??2"Adam/lstm_14/lstm_cell_14/kernel/m
=:;	@?2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
-:+?2 Adam/lstm_14/lstm_cell_14/bias/m
&:$@@2Adam/dense_12/kernel/v
 :@2Adam/dense_12/bias/v
&:$@ 2Adam/dense_13/kernel/v
 : 2Adam/dense_13/bias/v
&:$ 2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
3:1	~?2"Adam/lstm_12/lstm_cell_12/kernel/v
=:;	@?2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+?2 Adam/lstm_12/lstm_cell_12/bias/v
3:1	@?2"Adam/lstm_13/lstm_cell_13/kernel/v
>:<
??2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+?2 Adam/lstm_13/lstm_cell_13/bias/v
4:2
??2"Adam/lstm_14/lstm_cell_14/kernel/v
=:;	@?2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
-:+?2 Adam/lstm_14/lstm_cell_14/bias/v
?2?
 __inference__wrapped_model_86755?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
lstm_12_input?????????
~
?2?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89676
G__inference_sequential_4_layer_call_and_return_conditional_losses_89716
G__inference_sequential_4_layer_call_and_return_conditional_losses_90856
G__inference_sequential_4_layer_call_and_return_conditional_losses_90384?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_4_layer_call_fn_90891
,__inference_sequential_4_layer_call_fn_89792
,__inference_sequential_4_layer_call_fn_89867
,__inference_sequential_4_layer_call_fn_90926?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91079
B__inference_lstm_12_layer_call_and_return_conditional_losses_91407
B__inference_lstm_12_layer_call_and_return_conditional_losses_91232
B__inference_lstm_12_layer_call_and_return_conditional_losses_91560?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_12_layer_call_fn_91571
'__inference_lstm_12_layer_call_fn_91254
'__inference_lstm_12_layer_call_fn_91243
'__inference_lstm_12_layer_call_fn_91582?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_13_layer_call_and_return_conditional_losses_91735
B__inference_lstm_13_layer_call_and_return_conditional_losses_92216
B__inference_lstm_13_layer_call_and_return_conditional_losses_92063
B__inference_lstm_13_layer_call_and_return_conditional_losses_91888?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_13_layer_call_fn_92227
'__inference_lstm_13_layer_call_fn_92238
'__inference_lstm_13_layer_call_fn_91899
'__inference_lstm_13_layer_call_fn_91910?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92391
B__inference_lstm_14_layer_call_and_return_conditional_losses_92719
B__inference_lstm_14_layer_call_and_return_conditional_losses_92544
B__inference_lstm_14_layer_call_and_return_conditional_losses_92872?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_14_layer_call_fn_92555
'__inference_lstm_14_layer_call_fn_92894
'__inference_lstm_14_layer_call_fn_92883
'__inference_lstm_14_layer_call_fn_92566?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_12_layer_call_and_return_conditional_losses_92905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_12_layer_call_fn_92914?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_13_layer_call_and_return_conditional_losses_92925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_13_layer_call_fn_92934?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_14_layer_call_and_return_conditional_losses_92945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_14_layer_call_fn_92954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_89912lstm_12_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_93020
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_92987?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_12_layer_call_fn_93037
,__inference_lstm_cell_12_layer_call_fn_93054?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93120
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93087?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_13_layer_call_fn_93137
,__inference_lstm_cell_13_layer_call_fn_93154?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93187
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93220?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_14_layer_call_fn_93254
,__inference_lstm_cell_14_layer_call_fn_93237?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_86755?6789:;<=> %&+,:?7
0?-
+?(
lstm_12_input?????????
~
? "3?0
.
dense_14"?
dense_14??????????
C__inference_dense_12_layer_call_and_return_conditional_losses_92905\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_dense_12_layer_call_fn_92914O /?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_dense_13_layer_call_and_return_conditional_losses_92925\%&/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_13_layer_call_fn_92934O%&/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_14_layer_call_and_return_conditional_losses_92945\+,/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_14_layer_call_fn_92954O+,/?,
%?"
 ?
inputs????????? 
? "???????????
B__inference_lstm_12_layer_call_and_return_conditional_losses_91079?678O?L
E?B
4?1
/?,
inputs/0??????????????????~

 
p

 
? "2?/
(?%
0??????????????????@
? ?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91232?678O?L
E?B
4?1
/?,
inputs/0??????????????????~

 
p 

 
? "2?/
(?%
0??????????????????@
? ?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91407q678??<
5?2
$?!
inputs?????????
~

 
p

 
? ")?&
?
0?????????
@
? ?
B__inference_lstm_12_layer_call_and_return_conditional_losses_91560q678??<
5?2
$?!
inputs?????????
~

 
p 

 
? ")?&
?
0?????????
@
? ?
'__inference_lstm_12_layer_call_fn_91243}678O?L
E?B
4?1
/?,
inputs/0??????????????????~

 
p

 
? "%?"??????????????????@?
'__inference_lstm_12_layer_call_fn_91254}678O?L
E?B
4?1
/?,
inputs/0??????????????????~

 
p 

 
? "%?"??????????????????@?
'__inference_lstm_12_layer_call_fn_91571d678??<
5?2
$?!
inputs?????????
~

 
p

 
? "??????????
@?
'__inference_lstm_12_layer_call_fn_91582d678??<
5?2
$?!
inputs?????????
~

 
p 

 
? "??????????
@?
B__inference_lstm_13_layer_call_and_return_conditional_losses_91735?9:;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_13_layer_call_and_return_conditional_losses_91888?9:;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "3?0
)?&
0???????????????????
? ?
B__inference_lstm_13_layer_call_and_return_conditional_losses_92063r9:;??<
5?2
$?!
inputs?????????
@

 
p

 
? "*?'
 ?
0?????????
?
? ?
B__inference_lstm_13_layer_call_and_return_conditional_losses_92216r9:;??<
5?2
$?!
inputs?????????
@

 
p 

 
? "*?'
 ?
0?????????
?
? ?
'__inference_lstm_13_layer_call_fn_91899~9:;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "&?#????????????????????
'__inference_lstm_13_layer_call_fn_91910~9:;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "&?#????????????????????
'__inference_lstm_13_layer_call_fn_92227e9:;??<
5?2
$?!
inputs?????????
@

 
p

 
? "??????????
??
'__inference_lstm_13_layer_call_fn_92238e9:;??<
5?2
$?!
inputs?????????
@

 
p 

 
? "??????????
??
B__inference_lstm_14_layer_call_and_return_conditional_losses_92391n<=>@?=
6?3
%?"
inputs?????????
?

 
p

 
? "%?"
?
0?????????@
? ?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92544n<=>@?=
6?3
%?"
inputs?????????
?

 
p 

 
? "%?"
?
0?????????@
? ?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92719~<=>P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"
?
0?????????@
? ?
B__inference_lstm_14_layer_call_and_return_conditional_losses_92872~<=>P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"
?
0?????????@
? ?
'__inference_lstm_14_layer_call_fn_92555a<=>@?=
6?3
%?"
inputs?????????
?

 
p

 
? "??????????@?
'__inference_lstm_14_layer_call_fn_92566a<=>@?=
6?3
%?"
inputs?????????
?

 
p 

 
? "??????????@?
'__inference_lstm_14_layer_call_fn_92883q<=>P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "??????????@?
'__inference_lstm_14_layer_call_fn_92894q<=>P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "??????????@?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_92987?678??}
v?s
 ?
inputs?????????~
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
G__inference_lstm_cell_12_layer_call_and_return_conditional_losses_93020?678??}
v?s
 ?
inputs?????????~
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
,__inference_lstm_cell_12_layer_call_fn_93037?678??}
v?s
 ?
inputs?????????~
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
,__inference_lstm_cell_12_layer_call_fn_93054?678??}
v?s
 ?
inputs?????????~
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93087?9:;??
x?u
 ?
inputs?????????@
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
G__inference_lstm_cell_13_layer_call_and_return_conditional_losses_93120?9:;??
x?u
 ?
inputs?????????@
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
,__inference_lstm_cell_13_layer_call_fn_93137?9:;??
x?u
 ?
inputs?????????@
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
,__inference_lstm_cell_13_layer_call_fn_93154?9:;??
x?u
 ?
inputs?????????@
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93187?<=>??~
w?t
!?
inputs??????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
G__inference_lstm_cell_14_layer_call_and_return_conditional_losses_93220?<=>??~
w?t
!?
inputs??????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
,__inference_lstm_cell_14_layer_call_fn_93237?<=>??~
w?t
!?
inputs??????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
,__inference_lstm_cell_14_layer_call_fn_93254?<=>??~
w?t
!?
inputs??????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89676|6789:;<=> %&+,B??
8?5
+?(
lstm_12_input?????????
~
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_89716|6789:;<=> %&+,B??
8?5
+?(
lstm_12_input?????????
~
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_90384u6789:;<=> %&+,;?8
1?.
$?!
inputs?????????
~
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_90856u6789:;<=> %&+,;?8
1?.
$?!
inputs?????????
~
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_4_layer_call_fn_89792o6789:;<=> %&+,B??
8?5
+?(
lstm_12_input?????????
~
p

 
? "???????????
,__inference_sequential_4_layer_call_fn_89867o6789:;<=> %&+,B??
8?5
+?(
lstm_12_input?????????
~
p 

 
? "???????????
,__inference_sequential_4_layer_call_fn_90891h6789:;<=> %&+,;?8
1?.
$?!
inputs?????????
~
p

 
? "???????????
,__inference_sequential_4_layer_call_fn_90926h6789:;<=> %&+,;?8
1?.
$?!
inputs?????????
~
p 

 
? "???????????
#__inference_signature_wrapper_89912?6789:;<=> %&+,K?H
? 
A?>
<
lstm_12_input+?(
lstm_12_input?????????
~"3?0
.
dense_14"?
dense_14?????????