ĹÖ
őŮ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02unknown8Ş
l
save_counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namesave_counter
e
 save_counter/Read/ReadVariableOpReadVariableOpsave_counter*
_output_shapes
: *
dtype0	
P
unused_resourcePlaceholder*
_output_shapes
: *
dtype0*
shape: 
R
unused_resource_1Placeholder*
_output_shapes
: *
dtype0*
shape: 

network/layer1/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namenetwork/layer1/conv/weights

/network/layer1/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer1/conv/weights*&
_output_shapes
: *
dtype0

"network/layer1/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"network/layer1/conv/BatchNorm/beta

6network/layer1/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp"network/layer1/conv/BatchNorm/beta*
_output_shapes
: *
dtype0
Ş
)network/layer1/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)network/layer1/conv/BatchNorm/moving_mean
Ł
=network/layer1/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp)network/layer1/conv/BatchNorm/moving_mean*
_output_shapes
: *
dtype0
˛
-network/layer1/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-network/layer1/conv/BatchNorm/moving_variance
Ť
Anetwork/layer1/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp-network/layer1/conv/BatchNorm/moving_variance*
_output_shapes
: *
dtype0
´
(network/layer2/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(network/layer2/sepconv/depthwise_weights
­
<network/layer2/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp(network/layer2/sepconv/depthwise_weights*&
_output_shapes
: *
dtype0
˘
%network/layer2/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%network/layer2/sepconv/BatchNorm/beta

9network/layer2/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp%network/layer2/sepconv/BatchNorm/beta*
_output_shapes
: *
dtype0
°
,network/layer2/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,network/layer2/sepconv/BatchNorm/moving_mean
Š
@network/layer2/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp,network/layer2/sepconv/BatchNorm/moving_mean*
_output_shapes
: *
dtype0
¸
0network/layer2/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20network/layer2/sepconv/BatchNorm/moving_variance
ą
Dnetwork/layer2/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp0network/layer2/sepconv/BatchNorm/moving_variance*
_output_shapes
: *
dtype0

network/layer3/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_namenetwork/layer3/conv/weights

/network/layer3/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer3/conv/weights*&
_output_shapes
: @*
dtype0

"network/layer3/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"network/layer3/conv/BatchNorm/beta

6network/layer3/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp"network/layer3/conv/BatchNorm/beta*
_output_shapes
:@*
dtype0
Ş
)network/layer3/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)network/layer3/conv/BatchNorm/moving_mean
Ł
=network/layer3/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp)network/layer3/conv/BatchNorm/moving_mean*
_output_shapes
:@*
dtype0
˛
-network/layer3/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-network/layer3/conv/BatchNorm/moving_variance
Ť
Anetwork/layer3/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp-network/layer3/conv/BatchNorm/moving_variance*
_output_shapes
:@*
dtype0
´
(network/layer4/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(network/layer4/sepconv/depthwise_weights
­
<network/layer4/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp(network/layer4/sepconv/depthwise_weights*&
_output_shapes
:@*
dtype0
˘
%network/layer4/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%network/layer4/sepconv/BatchNorm/beta

9network/layer4/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp%network/layer4/sepconv/BatchNorm/beta*
_output_shapes
:@*
dtype0
°
,network/layer4/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,network/layer4/sepconv/BatchNorm/moving_mean
Š
@network/layer4/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp,network/layer4/sepconv/BatchNorm/moving_mean*
_output_shapes
:@*
dtype0
¸
0network/layer4/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20network/layer4/sepconv/BatchNorm/moving_variance
ą
Dnetwork/layer4/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp0network/layer4/sepconv/BatchNorm/moving_variance*
_output_shapes
:@*
dtype0

network/layer5/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namenetwork/layer5/conv/weights

/network/layer5/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer5/conv/weights*'
_output_shapes
:@*
dtype0

"network/layer5/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"network/layer5/conv/BatchNorm/beta

6network/layer5/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp"network/layer5/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ť
)network/layer5/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer5/conv/BatchNorm/moving_mean
¤
=network/layer5/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp)network/layer5/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ł
-network/layer5/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer5/conv/BatchNorm/moving_variance
Ź
Anetwork/layer5/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp-network/layer5/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ľ
(network/layer6/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(network/layer6/sepconv/depthwise_weights
Ž
<network/layer6/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp(network/layer6/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ł
%network/layer6/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%network/layer6/sepconv/BatchNorm/beta

9network/layer6/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp%network/layer6/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ą
,network/layer6/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,network/layer6/sepconv/BatchNorm/moving_mean
Ş
@network/layer6/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp,network/layer6/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
š
0network/layer6/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20network/layer6/sepconv/BatchNorm/moving_variance
˛
Dnetwork/layer6/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp0network/layer6/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer7/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namenetwork/layer7/conv/weights

/network/layer7/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer7/conv/weights*(
_output_shapes
:*
dtype0

"network/layer7/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"network/layer7/conv/BatchNorm/beta

6network/layer7/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp"network/layer7/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ť
)network/layer7/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer7/conv/BatchNorm/moving_mean
¤
=network/layer7/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp)network/layer7/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ł
-network/layer7/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer7/conv/BatchNorm/moving_variance
Ź
Anetwork/layer7/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp-network/layer7/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ľ
(network/layer8/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(network/layer8/sepconv/depthwise_weights
Ž
<network/layer8/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp(network/layer8/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ł
%network/layer8/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%network/layer8/sepconv/BatchNorm/beta

9network/layer8/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp%network/layer8/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ą
,network/layer8/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,network/layer8/sepconv/BatchNorm/moving_mean
Ş
@network/layer8/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp,network/layer8/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
š
0network/layer8/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20network/layer8/sepconv/BatchNorm/moving_variance
˛
Dnetwork/layer8/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp0network/layer8/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer9/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namenetwork/layer9/conv/weights

/network/layer9/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer9/conv/weights*(
_output_shapes
:*
dtype0

"network/layer9/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"network/layer9/conv/BatchNorm/beta

6network/layer9/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp"network/layer9/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ť
)network/layer9/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer9/conv/BatchNorm/moving_mean
¤
=network/layer9/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp)network/layer9/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ł
-network/layer9/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer9/conv/BatchNorm/moving_variance
Ź
Anetwork/layer9/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp-network/layer9/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer10/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer10/sepconv/depthwise_weights
°
=network/layer10/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer10/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer10/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer10/sepconv/BatchNorm/beta

:network/layer10/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer10/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer10/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer10/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer10/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer10/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer10/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer10/sepconv/BatchNorm/moving_variance
´
Enetwork/layer10/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer10/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer11/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer11/conv/weights

0network/layer11/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer11/conv/weights*(
_output_shapes
:*
dtype0

#network/layer11/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer11/conv/BatchNorm/beta

7network/layer11/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer11/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer11/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer11/conv/BatchNorm/moving_mean
Ś
>network/layer11/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer11/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer11/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer11/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer11/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer11/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer12/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer12/sepconv/depthwise_weights
°
=network/layer12/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer12/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer12/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer12/sepconv/BatchNorm/beta

:network/layer12/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer12/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer12/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer12/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer12/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer12/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer12/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer12/sepconv/BatchNorm/moving_variance
´
Enetwork/layer12/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer12/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer13/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer13/conv/weights

0network/layer13/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer13/conv/weights*(
_output_shapes
:*
dtype0

#network/layer13/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer13/conv/BatchNorm/beta

7network/layer13/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer13/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer13/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer13/conv/BatchNorm/moving_mean
Ś
>network/layer13/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer13/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer13/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer13/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer13/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer13/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer14/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer14/sepconv/depthwise_weights
°
=network/layer14/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer14/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer14/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer14/sepconv/BatchNorm/beta

:network/layer14/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer14/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer14/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer14/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer14/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer14/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer14/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer14/sepconv/BatchNorm/moving_variance
´
Enetwork/layer14/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer14/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer15/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer15/conv/weights

0network/layer15/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer15/conv/weights*(
_output_shapes
:*
dtype0

#network/layer15/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer15/conv/BatchNorm/beta

7network/layer15/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer15/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer15/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer15/conv/BatchNorm/moving_mean
Ś
>network/layer15/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer15/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer15/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer15/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer15/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer15/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer16/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer16/sepconv/depthwise_weights
°
=network/layer16/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer16/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer16/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer16/sepconv/BatchNorm/beta

:network/layer16/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer16/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer16/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer16/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer16/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer16/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer16/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer16/sepconv/BatchNorm/moving_variance
´
Enetwork/layer16/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer16/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer17/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer17/conv/weights

0network/layer17/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer17/conv/weights*(
_output_shapes
:*
dtype0

#network/layer17/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer17/conv/BatchNorm/beta

7network/layer17/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer17/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer17/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer17/conv/BatchNorm/moving_mean
Ś
>network/layer17/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer17/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer17/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer17/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer17/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer17/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer18/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer18/sepconv/depthwise_weights
°
=network/layer18/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer18/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer18/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer18/sepconv/BatchNorm/beta

:network/layer18/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer18/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer18/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer18/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer18/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer18/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer18/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer18/sepconv/BatchNorm/moving_variance
´
Enetwork/layer18/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer18/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer19/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer19/conv/weights

0network/layer19/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer19/conv/weights*(
_output_shapes
:*
dtype0

#network/layer19/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer19/conv/BatchNorm/beta

7network/layer19/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer19/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer19/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer19/conv/BatchNorm/moving_mean
Ś
>network/layer19/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer19/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer19/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer19/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer19/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer19/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer20/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer20/sepconv/depthwise_weights
°
=network/layer20/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer20/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer20/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer20/sepconv/BatchNorm/beta

:network/layer20/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer20/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer20/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer20/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer20/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer20/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer20/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer20/sepconv/BatchNorm/moving_variance
´
Enetwork/layer20/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer20/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer21/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer21/conv/weights

0network/layer21/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer21/conv/weights*(
_output_shapes
:*
dtype0

#network/layer21/conv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#network/layer21/conv/BatchNorm/beta

7network/layer21/conv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#network/layer21/conv/BatchNorm/beta*
_output_shapes	
:*
dtype0
­
*network/layer21/conv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network/layer21/conv/BatchNorm/moving_mean
Ś
>network/layer21/conv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*network/layer21/conv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ľ
.network/layer21/conv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.network/layer21/conv/BatchNorm/moving_variance
Ž
Bnetwork/layer21/conv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.network/layer21/conv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ˇ
)network/layer22/sepconv/depthwise_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network/layer22/sepconv/depthwise_weights
°
=network/layer22/sepconv/depthwise_weights/Read/ReadVariableOpReadVariableOp)network/layer22/sepconv/depthwise_weights*'
_output_shapes
:*
dtype0
Ľ
&network/layer22/sepconv/BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network/layer22/sepconv/BatchNorm/beta

:network/layer22/sepconv/BatchNorm/beta/Read/ReadVariableOpReadVariableOp&network/layer22/sepconv/BatchNorm/beta*
_output_shapes	
:*
dtype0
ł
-network/layer22/sepconv/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network/layer22/sepconv/BatchNorm/moving_mean
Ź
Anetwork/layer22/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp-network/layer22/sepconv/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
ť
1network/layer22/sepconv/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31network/layer22/sepconv/BatchNorm/moving_variance
´
Enetwork/layer22/sepconv/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp1network/layer22/sepconv/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0

network/layer23/conv/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenetwork/layer23/conv/weights

0network/layer23/conv/weights/Read/ReadVariableOpReadVariableOpnetwork/layer23/conv/weights*(
_output_shapes
:*
dtype0

network/layer23/conv/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namenetwork/layer23/conv/biases

/network/layer23/conv/biases/Read/ReadVariableOpReadVariableOpnetwork/layer23/conv/biases*
_output_shapes	
:*
dtype0

network/layer25/fc/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
`*+
shared_namenetwork/layer25/fc/weights

.network/layer25/fc/weights/Read/ReadVariableOpReadVariableOpnetwork/layer25/fc/weights* 
_output_shapes
:
`*
dtype0

network/layer25/fc/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namenetwork/layer25/fc/biases

-network/layer25/fc/biases/Read/ReadVariableOpReadVariableOpnetwork/layer25/fc/biases*
_output_shapes	
:*
dtype0

network/layer28/fc/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
`*+
shared_namenetwork/layer28/fc/weights

.network/layer28/fc/weights/Read/ReadVariableOpReadVariableOpnetwork/layer28/fc/weights* 
_output_shapes
:
`*
dtype0

network/layer28/fc/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:`**
shared_namenetwork/layer28/fc/biases

-network/layer28/fc/biases/Read/ReadVariableOpReadVariableOpnetwork/layer28/fc/biases*
_output_shapes	
:`*
dtype0
Ę!
PartitionedCallPartitionedCallnetwork/layer1/conv/weights"network/layer1/conv/BatchNorm/beta)network/layer1/conv/BatchNorm/moving_mean-network/layer1/conv/BatchNorm/moving_variance(network/layer2/sepconv/depthwise_weights%network/layer2/sepconv/BatchNorm/beta,network/layer2/sepconv/BatchNorm/moving_mean0network/layer2/sepconv/BatchNorm/moving_variancenetwork/layer3/conv/weights"network/layer3/conv/BatchNorm/beta)network/layer3/conv/BatchNorm/moving_mean-network/layer3/conv/BatchNorm/moving_variance(network/layer4/sepconv/depthwise_weights%network/layer4/sepconv/BatchNorm/beta,network/layer4/sepconv/BatchNorm/moving_mean0network/layer4/sepconv/BatchNorm/moving_variancenetwork/layer5/conv/weights"network/layer5/conv/BatchNorm/beta)network/layer5/conv/BatchNorm/moving_mean-network/layer5/conv/BatchNorm/moving_variance(network/layer6/sepconv/depthwise_weights%network/layer6/sepconv/BatchNorm/beta,network/layer6/sepconv/BatchNorm/moving_mean0network/layer6/sepconv/BatchNorm/moving_variancenetwork/layer7/conv/weights"network/layer7/conv/BatchNorm/beta)network/layer7/conv/BatchNorm/moving_mean-network/layer7/conv/BatchNorm/moving_variance(network/layer8/sepconv/depthwise_weights%network/layer8/sepconv/BatchNorm/beta,network/layer8/sepconv/BatchNorm/moving_mean0network/layer8/sepconv/BatchNorm/moving_variancenetwork/layer9/conv/weights"network/layer9/conv/BatchNorm/beta)network/layer9/conv/BatchNorm/moving_mean-network/layer9/conv/BatchNorm/moving_variance)network/layer10/sepconv/depthwise_weights&network/layer10/sepconv/BatchNorm/beta-network/layer10/sepconv/BatchNorm/moving_mean1network/layer10/sepconv/BatchNorm/moving_variancenetwork/layer11/conv/weights#network/layer11/conv/BatchNorm/beta*network/layer11/conv/BatchNorm/moving_mean.network/layer11/conv/BatchNorm/moving_variance)network/layer12/sepconv/depthwise_weights&network/layer12/sepconv/BatchNorm/beta-network/layer12/sepconv/BatchNorm/moving_mean1network/layer12/sepconv/BatchNorm/moving_variancenetwork/layer13/conv/weights#network/layer13/conv/BatchNorm/beta*network/layer13/conv/BatchNorm/moving_mean.network/layer13/conv/BatchNorm/moving_variance)network/layer14/sepconv/depthwise_weights&network/layer14/sepconv/BatchNorm/beta-network/layer14/sepconv/BatchNorm/moving_mean1network/layer14/sepconv/BatchNorm/moving_variancenetwork/layer15/conv/weights#network/layer15/conv/BatchNorm/beta*network/layer15/conv/BatchNorm/moving_mean.network/layer15/conv/BatchNorm/moving_variance)network/layer16/sepconv/depthwise_weights&network/layer16/sepconv/BatchNorm/beta-network/layer16/sepconv/BatchNorm/moving_mean1network/layer16/sepconv/BatchNorm/moving_variancenetwork/layer17/conv/weights#network/layer17/conv/BatchNorm/beta*network/layer17/conv/BatchNorm/moving_mean.network/layer17/conv/BatchNorm/moving_variance)network/layer18/sepconv/depthwise_weights&network/layer18/sepconv/BatchNorm/beta-network/layer18/sepconv/BatchNorm/moving_mean1network/layer18/sepconv/BatchNorm/moving_variancenetwork/layer19/conv/weights#network/layer19/conv/BatchNorm/beta*network/layer19/conv/BatchNorm/moving_mean.network/layer19/conv/BatchNorm/moving_variance)network/layer20/sepconv/depthwise_weights&network/layer20/sepconv/BatchNorm/beta-network/layer20/sepconv/BatchNorm/moving_mean1network/layer20/sepconv/BatchNorm/moving_variancenetwork/layer21/conv/weights#network/layer21/conv/BatchNorm/beta*network/layer21/conv/BatchNorm/moving_mean.network/layer21/conv/BatchNorm/moving_variance)network/layer22/sepconv/depthwise_weights&network/layer22/sepconv/BatchNorm/beta-network/layer22/sepconv/BatchNorm/moving_mean1network/layer22/sepconv/BatchNorm/moving_variancenetwork/layer23/conv/weightsnetwork/layer23/conv/biasesnetwork/layer25/fc/weightsnetwork/layer25/fc/biasesnetwork/layer28/fc/weightsnetwork/layer28/fc/biases*i
Tinb
`2^*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: *
_read_only_resource_inputsb
`^ 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]*-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_<lambda>_7068

NoOpNoOp^PartitionedCall
łS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*îR
valueäRBáR BÚR
\
trill_module
	variables
trainable_variables
save_counter

signatures
A
initializer
asset_paths

signatures
	variables
ć
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93
IG
VARIABLE_VALUEsave_counter'save_counter/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
WU
VARIABLE_VALUEnetwork/layer1/conv/weights&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"network/layer1/conv/BatchNorm/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)network/layer1/conv/BatchNorm/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-network/layer1/conv/BatchNorm/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(network/layer2/sepconv/depthwise_weights&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%network/layer2/sepconv/BatchNorm/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE,network/layer2/sepconv/BatchNorm/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0network/layer2/sepconv/BatchNorm/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnetwork/layer3/conv/weights&variables/8/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"network/layer3/conv/BatchNorm/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer3/conv/BatchNorm/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer3/conv/BatchNorm/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(network/layer4/sepconv/depthwise_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%network/layer4/sepconv/BatchNorm/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network/layer4/sepconv/BatchNorm/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0network/layer4/sepconv/BatchNorm/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnetwork/layer5/conv/weights'variables/16/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"network/layer5/conv/BatchNorm/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer5/conv/BatchNorm/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer5/conv/BatchNorm/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(network/layer6/sepconv/depthwise_weights'variables/20/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%network/layer6/sepconv/BatchNorm/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network/layer6/sepconv/BatchNorm/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0network/layer6/sepconv/BatchNorm/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnetwork/layer7/conv/weights'variables/24/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"network/layer7/conv/BatchNorm/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer7/conv/BatchNorm/moving_mean'variables/26/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer7/conv/BatchNorm/moving_variance'variables/27/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(network/layer8/sepconv/depthwise_weights'variables/28/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%network/layer8/sepconv/BatchNorm/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network/layer8/sepconv/BatchNorm/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0network/layer8/sepconv/BatchNorm/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnetwork/layer9/conv/weights'variables/32/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"network/layer9/conv/BatchNorm/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer9/conv/BatchNorm/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer9/conv/BatchNorm/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer10/sepconv/depthwise_weights'variables/36/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer10/sepconv/BatchNorm/beta'variables/37/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer10/sepconv/BatchNorm/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer10/sepconv/BatchNorm/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer11/conv/weights'variables/40/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer11/conv/BatchNorm/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer11/conv/BatchNorm/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer11/conv/BatchNorm/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer12/sepconv/depthwise_weights'variables/44/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer12/sepconv/BatchNorm/beta'variables/45/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer12/sepconv/BatchNorm/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer12/sepconv/BatchNorm/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer13/conv/weights'variables/48/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer13/conv/BatchNorm/beta'variables/49/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer13/conv/BatchNorm/moving_mean'variables/50/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer13/conv/BatchNorm/moving_variance'variables/51/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer14/sepconv/depthwise_weights'variables/52/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer14/sepconv/BatchNorm/beta'variables/53/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer14/sepconv/BatchNorm/moving_mean'variables/54/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer14/sepconv/BatchNorm/moving_variance'variables/55/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer15/conv/weights'variables/56/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer15/conv/BatchNorm/beta'variables/57/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer15/conv/BatchNorm/moving_mean'variables/58/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer15/conv/BatchNorm/moving_variance'variables/59/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer16/sepconv/depthwise_weights'variables/60/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer16/sepconv/BatchNorm/beta'variables/61/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer16/sepconv/BatchNorm/moving_mean'variables/62/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer16/sepconv/BatchNorm/moving_variance'variables/63/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer17/conv/weights'variables/64/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer17/conv/BatchNorm/beta'variables/65/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer17/conv/BatchNorm/moving_mean'variables/66/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer17/conv/BatchNorm/moving_variance'variables/67/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer18/sepconv/depthwise_weights'variables/68/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer18/sepconv/BatchNorm/beta'variables/69/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer18/sepconv/BatchNorm/moving_mean'variables/70/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer18/sepconv/BatchNorm/moving_variance'variables/71/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer19/conv/weights'variables/72/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer19/conv/BatchNorm/beta'variables/73/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer19/conv/BatchNorm/moving_mean'variables/74/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer19/conv/BatchNorm/moving_variance'variables/75/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer20/sepconv/depthwise_weights'variables/76/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer20/sepconv/BatchNorm/beta'variables/77/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer20/sepconv/BatchNorm/moving_mean'variables/78/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer20/sepconv/BatchNorm/moving_variance'variables/79/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer21/conv/weights'variables/80/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#network/layer21/conv/BatchNorm/beta'variables/81/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network/layer21/conv/BatchNorm/moving_mean'variables/82/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.network/layer21/conv/BatchNorm/moving_variance'variables/83/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)network/layer22/sepconv/depthwise_weights'variables/84/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&network/layer22/sepconv/BatchNorm/beta'variables/85/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network/layer22/sepconv/BatchNorm/moving_mean'variables/86/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1network/layer22/sepconv/BatchNorm/moving_variance'variables/87/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnetwork/layer23/conv/weights'variables/88/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnetwork/layer23/conv/biases'variables/89/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnetwork/layer25/fc/weights'variables/90/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnetwork/layer25/fc/biases'variables/91/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnetwork/layer28/fc/weights'variables/92/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnetwork/layer28/fc/biases'variables/93/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
0
StatefulPartitionedCallStatefulPartitionedCallsaver_filename save_counter/Read/ReadVariableOp/network/layer1/conv/weights/Read/ReadVariableOp6network/layer1/conv/BatchNorm/beta/Read/ReadVariableOp=network/layer1/conv/BatchNorm/moving_mean/Read/ReadVariableOpAnetwork/layer1/conv/BatchNorm/moving_variance/Read/ReadVariableOp<network/layer2/sepconv/depthwise_weights/Read/ReadVariableOp9network/layer2/sepconv/BatchNorm/beta/Read/ReadVariableOp@network/layer2/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpDnetwork/layer2/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp/network/layer3/conv/weights/Read/ReadVariableOp6network/layer3/conv/BatchNorm/beta/Read/ReadVariableOp=network/layer3/conv/BatchNorm/moving_mean/Read/ReadVariableOpAnetwork/layer3/conv/BatchNorm/moving_variance/Read/ReadVariableOp<network/layer4/sepconv/depthwise_weights/Read/ReadVariableOp9network/layer4/sepconv/BatchNorm/beta/Read/ReadVariableOp@network/layer4/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpDnetwork/layer4/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp/network/layer5/conv/weights/Read/ReadVariableOp6network/layer5/conv/BatchNorm/beta/Read/ReadVariableOp=network/layer5/conv/BatchNorm/moving_mean/Read/ReadVariableOpAnetwork/layer5/conv/BatchNorm/moving_variance/Read/ReadVariableOp<network/layer6/sepconv/depthwise_weights/Read/ReadVariableOp9network/layer6/sepconv/BatchNorm/beta/Read/ReadVariableOp@network/layer6/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpDnetwork/layer6/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp/network/layer7/conv/weights/Read/ReadVariableOp6network/layer7/conv/BatchNorm/beta/Read/ReadVariableOp=network/layer7/conv/BatchNorm/moving_mean/Read/ReadVariableOpAnetwork/layer7/conv/BatchNorm/moving_variance/Read/ReadVariableOp<network/layer8/sepconv/depthwise_weights/Read/ReadVariableOp9network/layer8/sepconv/BatchNorm/beta/Read/ReadVariableOp@network/layer8/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpDnetwork/layer8/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp/network/layer9/conv/weights/Read/ReadVariableOp6network/layer9/conv/BatchNorm/beta/Read/ReadVariableOp=network/layer9/conv/BatchNorm/moving_mean/Read/ReadVariableOpAnetwork/layer9/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer10/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer10/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer10/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer10/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer11/conv/weights/Read/ReadVariableOp7network/layer11/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer11/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer11/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer12/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer12/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer12/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer12/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer13/conv/weights/Read/ReadVariableOp7network/layer13/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer13/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer13/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer14/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer14/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer14/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer14/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer15/conv/weights/Read/ReadVariableOp7network/layer15/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer15/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer15/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer16/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer16/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer16/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer16/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer17/conv/weights/Read/ReadVariableOp7network/layer17/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer17/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer17/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer18/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer18/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer18/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer18/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer19/conv/weights/Read/ReadVariableOp7network/layer19/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer19/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer19/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer20/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer20/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer20/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer20/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer21/conv/weights/Read/ReadVariableOp7network/layer21/conv/BatchNorm/beta/Read/ReadVariableOp>network/layer21/conv/BatchNorm/moving_mean/Read/ReadVariableOpBnetwork/layer21/conv/BatchNorm/moving_variance/Read/ReadVariableOp=network/layer22/sepconv/depthwise_weights/Read/ReadVariableOp:network/layer22/sepconv/BatchNorm/beta/Read/ReadVariableOpAnetwork/layer22/sepconv/BatchNorm/moving_mean/Read/ReadVariableOpEnetwork/layer22/sepconv/BatchNorm/moving_variance/Read/ReadVariableOp0network/layer23/conv/weights/Read/ReadVariableOp/network/layer23/conv/biases/Read/ReadVariableOp.network/layer25/fc/weights/Read/ReadVariableOp-network/layer25/fc/biases/Read/ReadVariableOp.network/layer28/fc/weights/Read/ReadVariableOp-network/layer28/fc/biases/Read/ReadVariableOpConst*l
Tine
c2a	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_7377
˘!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamesave_counternetwork/layer1/conv/weights"network/layer1/conv/BatchNorm/beta)network/layer1/conv/BatchNorm/moving_mean-network/layer1/conv/BatchNorm/moving_variance(network/layer2/sepconv/depthwise_weights%network/layer2/sepconv/BatchNorm/beta,network/layer2/sepconv/BatchNorm/moving_mean0network/layer2/sepconv/BatchNorm/moving_variancenetwork/layer3/conv/weights"network/layer3/conv/BatchNorm/beta)network/layer3/conv/BatchNorm/moving_mean-network/layer3/conv/BatchNorm/moving_variance(network/layer4/sepconv/depthwise_weights%network/layer4/sepconv/BatchNorm/beta,network/layer4/sepconv/BatchNorm/moving_mean0network/layer4/sepconv/BatchNorm/moving_variancenetwork/layer5/conv/weights"network/layer5/conv/BatchNorm/beta)network/layer5/conv/BatchNorm/moving_mean-network/layer5/conv/BatchNorm/moving_variance(network/layer6/sepconv/depthwise_weights%network/layer6/sepconv/BatchNorm/beta,network/layer6/sepconv/BatchNorm/moving_mean0network/layer6/sepconv/BatchNorm/moving_variancenetwork/layer7/conv/weights"network/layer7/conv/BatchNorm/beta)network/layer7/conv/BatchNorm/moving_mean-network/layer7/conv/BatchNorm/moving_variance(network/layer8/sepconv/depthwise_weights%network/layer8/sepconv/BatchNorm/beta,network/layer8/sepconv/BatchNorm/moving_mean0network/layer8/sepconv/BatchNorm/moving_variancenetwork/layer9/conv/weights"network/layer9/conv/BatchNorm/beta)network/layer9/conv/BatchNorm/moving_mean-network/layer9/conv/BatchNorm/moving_variance)network/layer10/sepconv/depthwise_weights&network/layer10/sepconv/BatchNorm/beta-network/layer10/sepconv/BatchNorm/moving_mean1network/layer10/sepconv/BatchNorm/moving_variancenetwork/layer11/conv/weights#network/layer11/conv/BatchNorm/beta*network/layer11/conv/BatchNorm/moving_mean.network/layer11/conv/BatchNorm/moving_variance)network/layer12/sepconv/depthwise_weights&network/layer12/sepconv/BatchNorm/beta-network/layer12/sepconv/BatchNorm/moving_mean1network/layer12/sepconv/BatchNorm/moving_variancenetwork/layer13/conv/weights#network/layer13/conv/BatchNorm/beta*network/layer13/conv/BatchNorm/moving_mean.network/layer13/conv/BatchNorm/moving_variance)network/layer14/sepconv/depthwise_weights&network/layer14/sepconv/BatchNorm/beta-network/layer14/sepconv/BatchNorm/moving_mean1network/layer14/sepconv/BatchNorm/moving_variancenetwork/layer15/conv/weights#network/layer15/conv/BatchNorm/beta*network/layer15/conv/BatchNorm/moving_mean.network/layer15/conv/BatchNorm/moving_variance)network/layer16/sepconv/depthwise_weights&network/layer16/sepconv/BatchNorm/beta-network/layer16/sepconv/BatchNorm/moving_mean1network/layer16/sepconv/BatchNorm/moving_variancenetwork/layer17/conv/weights#network/layer17/conv/BatchNorm/beta*network/layer17/conv/BatchNorm/moving_mean.network/layer17/conv/BatchNorm/moving_variance)network/layer18/sepconv/depthwise_weights&network/layer18/sepconv/BatchNorm/beta-network/layer18/sepconv/BatchNorm/moving_mean1network/layer18/sepconv/BatchNorm/moving_variancenetwork/layer19/conv/weights#network/layer19/conv/BatchNorm/beta*network/layer19/conv/BatchNorm/moving_mean.network/layer19/conv/BatchNorm/moving_variance)network/layer20/sepconv/depthwise_weights&network/layer20/sepconv/BatchNorm/beta-network/layer20/sepconv/BatchNorm/moving_mean1network/layer20/sepconv/BatchNorm/moving_variancenetwork/layer21/conv/weights#network/layer21/conv/BatchNorm/beta*network/layer21/conv/BatchNorm/moving_mean.network/layer21/conv/BatchNorm/moving_variance)network/layer22/sepconv/depthwise_weights&network/layer22/sepconv/BatchNorm/beta-network/layer22/sepconv/BatchNorm/moving_mean1network/layer22/sepconv/BatchNorm/moving_variancenetwork/layer23/conv/weightsnetwork/layer23/conv/biasesnetwork/layer25/fc/weightsnetwork/layer25/fc/biasesnetwork/layer28/fc/weightsnetwork/layer28/fc/biases*k
Tind
b2`*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_7672ˇż

)
__inference__creator_6675
identityc
unused_resourcePlaceholder*
_output_shapes
: *
dtype0*
shape: 2
unused_resource[
IdentityIdentityunused_resource:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
ŕ
U
cond_false_4242
cond_placeholder
cond_identity_truediv
cond_identityo
cond/IdentityIdentitycond_identity_truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
9
Ö
__inference___call___6670
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/Assert§
PartitionedCallPartitionedCallsamples*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_27862
PartitionedCallV
ShapeShapePartitionedCall:output:0*
T0*
_output_shapes
:2
Shapes
CastCastPartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
CastŤ
StatefulPartitionedCallStatefulPartitionedCallCast:y:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallŽ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate
 

map_while_cond_5907$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_5907___redundant_placeholder0
map_while_identity

map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*!
_input_shapes
: : : : : :: 
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
: :

_output_shapes
: :

_output_shapes
:

F
cond_true_2357
cond_sub_size

cond_pad_x
cond_identity[

cond/sub/xConst*
_output_shapes
: *
dtype0*
value
B :}2

cond/sub/x`
cond/subSubcond/sub/x:output:0cond_sub_size*
T0*
_output_shapes
: 2

cond/subp
cond/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Pad/paddings/0/0
cond/Pad/paddings/0Packcond/Pad/paddings/0/0:output:0cond/sub:z:0*
N*
T0*
_output_shapes
:2
cond/Pad/paddings/0~
cond/Pad/paddingsPackcond/Pad/paddings/0:output:0*
N*
T0*
_output_shapes

:2
cond/Pad/paddingsq
cond/PadPad
cond_pad_xcond/Pad/paddings:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/Padk
cond/IdentityIdentitycond/Pad:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
đ@
 __inference__traced_restore_7672
file_prefix!
assignvariableop_save_counter2
.assignvariableop_1_network_layer1_conv_weights9
5assignvariableop_2_network_layer1_conv_batchnorm_beta@
<assignvariableop_3_network_layer1_conv_batchnorm_moving_meanD
@assignvariableop_4_network_layer1_conv_batchnorm_moving_variance?
;assignvariableop_5_network_layer2_sepconv_depthwise_weights<
8assignvariableop_6_network_layer2_sepconv_batchnorm_betaC
?assignvariableop_7_network_layer2_sepconv_batchnorm_moving_meanG
Cassignvariableop_8_network_layer2_sepconv_batchnorm_moving_variance2
.assignvariableop_9_network_layer3_conv_weights:
6assignvariableop_10_network_layer3_conv_batchnorm_betaA
=assignvariableop_11_network_layer3_conv_batchnorm_moving_meanE
Aassignvariableop_12_network_layer3_conv_batchnorm_moving_variance@
<assignvariableop_13_network_layer4_sepconv_depthwise_weights=
9assignvariableop_14_network_layer4_sepconv_batchnorm_betaD
@assignvariableop_15_network_layer4_sepconv_batchnorm_moving_meanH
Dassignvariableop_16_network_layer4_sepconv_batchnorm_moving_variance3
/assignvariableop_17_network_layer5_conv_weights:
6assignvariableop_18_network_layer5_conv_batchnorm_betaA
=assignvariableop_19_network_layer5_conv_batchnorm_moving_meanE
Aassignvariableop_20_network_layer5_conv_batchnorm_moving_variance@
<assignvariableop_21_network_layer6_sepconv_depthwise_weights=
9assignvariableop_22_network_layer6_sepconv_batchnorm_betaD
@assignvariableop_23_network_layer6_sepconv_batchnorm_moving_meanH
Dassignvariableop_24_network_layer6_sepconv_batchnorm_moving_variance3
/assignvariableop_25_network_layer7_conv_weights:
6assignvariableop_26_network_layer7_conv_batchnorm_betaA
=assignvariableop_27_network_layer7_conv_batchnorm_moving_meanE
Aassignvariableop_28_network_layer7_conv_batchnorm_moving_variance@
<assignvariableop_29_network_layer8_sepconv_depthwise_weights=
9assignvariableop_30_network_layer8_sepconv_batchnorm_betaD
@assignvariableop_31_network_layer8_sepconv_batchnorm_moving_meanH
Dassignvariableop_32_network_layer8_sepconv_batchnorm_moving_variance3
/assignvariableop_33_network_layer9_conv_weights:
6assignvariableop_34_network_layer9_conv_batchnorm_betaA
=assignvariableop_35_network_layer9_conv_batchnorm_moving_meanE
Aassignvariableop_36_network_layer9_conv_batchnorm_moving_varianceA
=assignvariableop_37_network_layer10_sepconv_depthwise_weights>
:assignvariableop_38_network_layer10_sepconv_batchnorm_betaE
Aassignvariableop_39_network_layer10_sepconv_batchnorm_moving_meanI
Eassignvariableop_40_network_layer10_sepconv_batchnorm_moving_variance4
0assignvariableop_41_network_layer11_conv_weights;
7assignvariableop_42_network_layer11_conv_batchnorm_betaB
>assignvariableop_43_network_layer11_conv_batchnorm_moving_meanF
Bassignvariableop_44_network_layer11_conv_batchnorm_moving_varianceA
=assignvariableop_45_network_layer12_sepconv_depthwise_weights>
:assignvariableop_46_network_layer12_sepconv_batchnorm_betaE
Aassignvariableop_47_network_layer12_sepconv_batchnorm_moving_meanI
Eassignvariableop_48_network_layer12_sepconv_batchnorm_moving_variance4
0assignvariableop_49_network_layer13_conv_weights;
7assignvariableop_50_network_layer13_conv_batchnorm_betaB
>assignvariableop_51_network_layer13_conv_batchnorm_moving_meanF
Bassignvariableop_52_network_layer13_conv_batchnorm_moving_varianceA
=assignvariableop_53_network_layer14_sepconv_depthwise_weights>
:assignvariableop_54_network_layer14_sepconv_batchnorm_betaE
Aassignvariableop_55_network_layer14_sepconv_batchnorm_moving_meanI
Eassignvariableop_56_network_layer14_sepconv_batchnorm_moving_variance4
0assignvariableop_57_network_layer15_conv_weights;
7assignvariableop_58_network_layer15_conv_batchnorm_betaB
>assignvariableop_59_network_layer15_conv_batchnorm_moving_meanF
Bassignvariableop_60_network_layer15_conv_batchnorm_moving_varianceA
=assignvariableop_61_network_layer16_sepconv_depthwise_weights>
:assignvariableop_62_network_layer16_sepconv_batchnorm_betaE
Aassignvariableop_63_network_layer16_sepconv_batchnorm_moving_meanI
Eassignvariableop_64_network_layer16_sepconv_batchnorm_moving_variance4
0assignvariableop_65_network_layer17_conv_weights;
7assignvariableop_66_network_layer17_conv_batchnorm_betaB
>assignvariableop_67_network_layer17_conv_batchnorm_moving_meanF
Bassignvariableop_68_network_layer17_conv_batchnorm_moving_varianceA
=assignvariableop_69_network_layer18_sepconv_depthwise_weights>
:assignvariableop_70_network_layer18_sepconv_batchnorm_betaE
Aassignvariableop_71_network_layer18_sepconv_batchnorm_moving_meanI
Eassignvariableop_72_network_layer18_sepconv_batchnorm_moving_variance4
0assignvariableop_73_network_layer19_conv_weights;
7assignvariableop_74_network_layer19_conv_batchnorm_betaB
>assignvariableop_75_network_layer19_conv_batchnorm_moving_meanF
Bassignvariableop_76_network_layer19_conv_batchnorm_moving_varianceA
=assignvariableop_77_network_layer20_sepconv_depthwise_weights>
:assignvariableop_78_network_layer20_sepconv_batchnorm_betaE
Aassignvariableop_79_network_layer20_sepconv_batchnorm_moving_meanI
Eassignvariableop_80_network_layer20_sepconv_batchnorm_moving_variance4
0assignvariableop_81_network_layer21_conv_weights;
7assignvariableop_82_network_layer21_conv_batchnorm_betaB
>assignvariableop_83_network_layer21_conv_batchnorm_moving_meanF
Bassignvariableop_84_network_layer21_conv_batchnorm_moving_varianceA
=assignvariableop_85_network_layer22_sepconv_depthwise_weights>
:assignvariableop_86_network_layer22_sepconv_batchnorm_betaE
Aassignvariableop_87_network_layer22_sepconv_batchnorm_moving_meanI
Eassignvariableop_88_network_layer22_sepconv_batchnorm_moving_variance4
0assignvariableop_89_network_layer23_conv_weights3
/assignvariableop_90_network_layer23_conv_biases2
.assignvariableop_91_network_layer25_fc_weights1
-assignvariableop_92_network_layer25_fc_biases2
.assignvariableop_93_network_layer28_fc_weights1
-assignvariableop_94_network_layer28_fc_biases
identity_96˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_35˘AssignVariableOp_36˘AssignVariableOp_37˘AssignVariableOp_38˘AssignVariableOp_39˘AssignVariableOp_4˘AssignVariableOp_40˘AssignVariableOp_41˘AssignVariableOp_42˘AssignVariableOp_43˘AssignVariableOp_44˘AssignVariableOp_45˘AssignVariableOp_46˘AssignVariableOp_47˘AssignVariableOp_48˘AssignVariableOp_49˘AssignVariableOp_5˘AssignVariableOp_50˘AssignVariableOp_51˘AssignVariableOp_52˘AssignVariableOp_53˘AssignVariableOp_54˘AssignVariableOp_55˘AssignVariableOp_56˘AssignVariableOp_57˘AssignVariableOp_58˘AssignVariableOp_59˘AssignVariableOp_6˘AssignVariableOp_60˘AssignVariableOp_61˘AssignVariableOp_62˘AssignVariableOp_63˘AssignVariableOp_64˘AssignVariableOp_65˘AssignVariableOp_66˘AssignVariableOp_67˘AssignVariableOp_68˘AssignVariableOp_69˘AssignVariableOp_7˘AssignVariableOp_70˘AssignVariableOp_71˘AssignVariableOp_72˘AssignVariableOp_73˘AssignVariableOp_74˘AssignVariableOp_75˘AssignVariableOp_76˘AssignVariableOp_77˘AssignVariableOp_78˘AssignVariableOp_79˘AssignVariableOp_8˘AssignVariableOp_80˘AssignVariableOp_81˘AssignVariableOp_82˘AssignVariableOp_83˘AssignVariableOp_84˘AssignVariableOp_85˘AssignVariableOp_86˘AssignVariableOp_87˘AssignVariableOp_88˘AssignVariableOp_89˘AssignVariableOp_9˘AssignVariableOp_90˘AssignVariableOp_91˘AssignVariableOp_92˘AssignVariableOp_93˘AssignVariableOp_94Ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*ŕ
valueÖBÓ`B'save_counter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŃ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*Ő
valueËBČ`B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*n
dtypesd
b2`	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_save_counterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ł
AssignVariableOp_1AssignVariableOp.assignvariableop_1_network_layer1_conv_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ş
AssignVariableOp_2AssignVariableOp5assignvariableop_2_network_layer1_conv_batchnorm_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Á
AssignVariableOp_3AssignVariableOp<assignvariableop_3_network_layer1_conv_batchnorm_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ĺ
AssignVariableOp_4AssignVariableOp@assignvariableop_4_network_layer1_conv_batchnorm_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ŕ
AssignVariableOp_5AssignVariableOp;assignvariableop_5_network_layer2_sepconv_depthwise_weightsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6˝
AssignVariableOp_6AssignVariableOp8assignvariableop_6_network_layer2_sepconv_batchnorm_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ä
AssignVariableOp_7AssignVariableOp?assignvariableop_7_network_layer2_sepconv_batchnorm_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Č
AssignVariableOp_8AssignVariableOpCassignvariableop_8_network_layer2_sepconv_batchnorm_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ł
AssignVariableOp_9AssignVariableOp.assignvariableop_9_network_layer3_conv_weightsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ž
AssignVariableOp_10AssignVariableOp6assignvariableop_10_network_layer3_conv_batchnorm_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ĺ
AssignVariableOp_11AssignVariableOp=assignvariableop_11_network_layer3_conv_batchnorm_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12É
AssignVariableOp_12AssignVariableOpAassignvariableop_12_network_layer3_conv_batchnorm_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ä
AssignVariableOp_13AssignVariableOp<assignvariableop_13_network_layer4_sepconv_depthwise_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_network_layer4_sepconv_batchnorm_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Č
AssignVariableOp_15AssignVariableOp@assignvariableop_15_network_layer4_sepconv_batchnorm_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ě
AssignVariableOp_16AssignVariableOpDassignvariableop_16_network_layer4_sepconv_batchnorm_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ˇ
AssignVariableOp_17AssignVariableOp/assignvariableop_17_network_layer5_conv_weightsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ž
AssignVariableOp_18AssignVariableOp6assignvariableop_18_network_layer5_conv_batchnorm_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ĺ
AssignVariableOp_19AssignVariableOp=assignvariableop_19_network_layer5_conv_batchnorm_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20É
AssignVariableOp_20AssignVariableOpAassignvariableop_20_network_layer5_conv_batchnorm_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ä
AssignVariableOp_21AssignVariableOp<assignvariableop_21_network_layer6_sepconv_depthwise_weightsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Á
AssignVariableOp_22AssignVariableOp9assignvariableop_22_network_layer6_sepconv_batchnorm_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Č
AssignVariableOp_23AssignVariableOp@assignvariableop_23_network_layer6_sepconv_batchnorm_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ě
AssignVariableOp_24AssignVariableOpDassignvariableop_24_network_layer6_sepconv_batchnorm_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ˇ
AssignVariableOp_25AssignVariableOp/assignvariableop_25_network_layer7_conv_weightsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ž
AssignVariableOp_26AssignVariableOp6assignvariableop_26_network_layer7_conv_batchnorm_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ĺ
AssignVariableOp_27AssignVariableOp=assignvariableop_27_network_layer7_conv_batchnorm_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28É
AssignVariableOp_28AssignVariableOpAassignvariableop_28_network_layer7_conv_batchnorm_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ä
AssignVariableOp_29AssignVariableOp<assignvariableop_29_network_layer8_sepconv_depthwise_weightsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Á
AssignVariableOp_30AssignVariableOp9assignvariableop_30_network_layer8_sepconv_batchnorm_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Č
AssignVariableOp_31AssignVariableOp@assignvariableop_31_network_layer8_sepconv_batchnorm_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ě
AssignVariableOp_32AssignVariableOpDassignvariableop_32_network_layer8_sepconv_batchnorm_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ˇ
AssignVariableOp_33AssignVariableOp/assignvariableop_33_network_layer9_conv_weightsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ž
AssignVariableOp_34AssignVariableOp6assignvariableop_34_network_layer9_conv_batchnorm_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ĺ
AssignVariableOp_35AssignVariableOp=assignvariableop_35_network_layer9_conv_batchnorm_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36É
AssignVariableOp_36AssignVariableOpAassignvariableop_36_network_layer9_conv_batchnorm_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ĺ
AssignVariableOp_37AssignVariableOp=assignvariableop_37_network_layer10_sepconv_depthwise_weightsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Â
AssignVariableOp_38AssignVariableOp:assignvariableop_38_network_layer10_sepconv_batchnorm_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39É
AssignVariableOp_39AssignVariableOpAassignvariableop_39_network_layer10_sepconv_batchnorm_moving_meanIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Í
AssignVariableOp_40AssignVariableOpEassignvariableop_40_network_layer10_sepconv_batchnorm_moving_varianceIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¸
AssignVariableOp_41AssignVariableOp0assignvariableop_41_network_layer11_conv_weightsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ż
AssignVariableOp_42AssignVariableOp7assignvariableop_42_network_layer11_conv_batchnorm_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ć
AssignVariableOp_43AssignVariableOp>assignvariableop_43_network_layer11_conv_batchnorm_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ę
AssignVariableOp_44AssignVariableOpBassignvariableop_44_network_layer11_conv_batchnorm_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ĺ
AssignVariableOp_45AssignVariableOp=assignvariableop_45_network_layer12_sepconv_depthwise_weightsIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Â
AssignVariableOp_46AssignVariableOp:assignvariableop_46_network_layer12_sepconv_batchnorm_betaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47É
AssignVariableOp_47AssignVariableOpAassignvariableop_47_network_layer12_sepconv_batchnorm_moving_meanIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Í
AssignVariableOp_48AssignVariableOpEassignvariableop_48_network_layer12_sepconv_batchnorm_moving_varianceIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¸
AssignVariableOp_49AssignVariableOp0assignvariableop_49_network_layer13_conv_weightsIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ż
AssignVariableOp_50AssignVariableOp7assignvariableop_50_network_layer13_conv_batchnorm_betaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ć
AssignVariableOp_51AssignVariableOp>assignvariableop_51_network_layer13_conv_batchnorm_moving_meanIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ę
AssignVariableOp_52AssignVariableOpBassignvariableop_52_network_layer13_conv_batchnorm_moving_varianceIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ĺ
AssignVariableOp_53AssignVariableOp=assignvariableop_53_network_layer14_sepconv_depthwise_weightsIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Â
AssignVariableOp_54AssignVariableOp:assignvariableop_54_network_layer14_sepconv_batchnorm_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55É
AssignVariableOp_55AssignVariableOpAassignvariableop_55_network_layer14_sepconv_batchnorm_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Í
AssignVariableOp_56AssignVariableOpEassignvariableop_56_network_layer14_sepconv_batchnorm_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¸
AssignVariableOp_57AssignVariableOp0assignvariableop_57_network_layer15_conv_weightsIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ż
AssignVariableOp_58AssignVariableOp7assignvariableop_58_network_layer15_conv_batchnorm_betaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ć
AssignVariableOp_59AssignVariableOp>assignvariableop_59_network_layer15_conv_batchnorm_moving_meanIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ę
AssignVariableOp_60AssignVariableOpBassignvariableop_60_network_layer15_conv_batchnorm_moving_varianceIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ĺ
AssignVariableOp_61AssignVariableOp=assignvariableop_61_network_layer16_sepconv_depthwise_weightsIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Â
AssignVariableOp_62AssignVariableOp:assignvariableop_62_network_layer16_sepconv_batchnorm_betaIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63É
AssignVariableOp_63AssignVariableOpAassignvariableop_63_network_layer16_sepconv_batchnorm_moving_meanIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Í
AssignVariableOp_64AssignVariableOpEassignvariableop_64_network_layer16_sepconv_batchnorm_moving_varianceIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¸
AssignVariableOp_65AssignVariableOp0assignvariableop_65_network_layer17_conv_weightsIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66ż
AssignVariableOp_66AssignVariableOp7assignvariableop_66_network_layer17_conv_batchnorm_betaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ć
AssignVariableOp_67AssignVariableOp>assignvariableop_67_network_layer17_conv_batchnorm_moving_meanIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ę
AssignVariableOp_68AssignVariableOpBassignvariableop_68_network_layer17_conv_batchnorm_moving_varianceIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ĺ
AssignVariableOp_69AssignVariableOp=assignvariableop_69_network_layer18_sepconv_depthwise_weightsIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Â
AssignVariableOp_70AssignVariableOp:assignvariableop_70_network_layer18_sepconv_batchnorm_betaIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71É
AssignVariableOp_71AssignVariableOpAassignvariableop_71_network_layer18_sepconv_batchnorm_moving_meanIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Í
AssignVariableOp_72AssignVariableOpEassignvariableop_72_network_layer18_sepconv_batchnorm_moving_varianceIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¸
AssignVariableOp_73AssignVariableOp0assignvariableop_73_network_layer19_conv_weightsIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74ż
AssignVariableOp_74AssignVariableOp7assignvariableop_74_network_layer19_conv_batchnorm_betaIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ć
AssignVariableOp_75AssignVariableOp>assignvariableop_75_network_layer19_conv_batchnorm_moving_meanIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ę
AssignVariableOp_76AssignVariableOpBassignvariableop_76_network_layer19_conv_batchnorm_moving_varianceIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ĺ
AssignVariableOp_77AssignVariableOp=assignvariableop_77_network_layer20_sepconv_depthwise_weightsIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Â
AssignVariableOp_78AssignVariableOp:assignvariableop_78_network_layer20_sepconv_batchnorm_betaIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79É
AssignVariableOp_79AssignVariableOpAassignvariableop_79_network_layer20_sepconv_batchnorm_moving_meanIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Í
AssignVariableOp_80AssignVariableOpEassignvariableop_80_network_layer20_sepconv_batchnorm_moving_varianceIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81¸
AssignVariableOp_81AssignVariableOp0assignvariableop_81_network_layer21_conv_weightsIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82ż
AssignVariableOp_82AssignVariableOp7assignvariableop_82_network_layer21_conv_batchnorm_betaIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ć
AssignVariableOp_83AssignVariableOp>assignvariableop_83_network_layer21_conv_batchnorm_moving_meanIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Ę
AssignVariableOp_84AssignVariableOpBassignvariableop_84_network_layer21_conv_batchnorm_moving_varianceIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ĺ
AssignVariableOp_85AssignVariableOp=assignvariableop_85_network_layer22_sepconv_depthwise_weightsIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Â
AssignVariableOp_86AssignVariableOp:assignvariableop_86_network_layer22_sepconv_batchnorm_betaIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87É
AssignVariableOp_87AssignVariableOpAassignvariableop_87_network_layer22_sepconv_batchnorm_moving_meanIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88Í
AssignVariableOp_88AssignVariableOpEassignvariableop_88_network_layer22_sepconv_batchnorm_moving_varianceIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¸
AssignVariableOp_89AssignVariableOp0assignvariableop_89_network_layer23_conv_weightsIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90ˇ
AssignVariableOp_90AssignVariableOp/assignvariableop_90_network_layer23_conv_biasesIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91ś
AssignVariableOp_91AssignVariableOp.assignvariableop_91_network_layer25_fc_weightsIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92ľ
AssignVariableOp_92AssignVariableOp-assignvariableop_92_network_layer25_fc_biasesIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93ś
AssignVariableOp_93AssignVariableOp.assignvariableop_93_network_layer28_fc_weightsIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94ľ
AssignVariableOp_94AssignVariableOp-assignvariableop_94_network_layer28_fc_biasesIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_949
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_95Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_95ű
Identity_96IdentityIdentity_95:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94*
T0*
_output_shapes
: 2
Identity_96"#
identity_96Identity_96:output:0*
_input_shapes
ţ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_94:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ô
O
cond_false_2358
cond_placeholder
cond_identity_x
cond_identityi
cond/IdentityIdentitycond_identity_x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
;
$__inference__sample_to_features_2786
x
identity8
SizeSizex*
T0*
_output_shapes
: 2
SizeS
Less/yConst*
_output_shapes
: *
dtype0*
value
B :}2
Less/yU
LessLessSize:output:0Less/y:output:0*
T0*
_output_shapes
: 2
Lessľ
condStatelessIfLess:z:0Size:output:0x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *"
else_branchR
cond_false_2358*"
output_shapes
:˙˙˙˙˙˙˙˙˙*!
then_branchR
cond_true_23572
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityi
CastCastcond/Identity:output:0*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/frame_lengthe
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B : 2
stft/frame_stepe
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/fft_lengthm
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
stft/frame/axis\
stft/frame/ShapeShapeCast:y:0*
T0*
_output_shapes
:2
stft/frame/Shaped
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Rankr
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range/startr
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range/deltaĽ
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:2
stft/frame/range
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
stft/frame/strided_slice/stack
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 stft/frame/strided_slice/stack_1
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 stft/frame/strided_slice/stack_2¤
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slicef
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub/y}
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_1l
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/packed/1ł
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packedz
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/split/split_dimÔ
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*
T0*

Tlen0*"
_output_shapes
: :: *
	num_split2
stft/frame/splitw
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape{
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape_1
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
stft/frame/Reshaped
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Sizeh
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Size_1
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_2
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: 2
stft/frame/floordivf
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/add/x~
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft/frame/addn
stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Maximum/x
stft/frame/MaximumMaximumstft/frame/Maximum/x:output:0stft/frame/add:z:0*
T0*
_output_shapes
: 2
stft/frame/Maximumn
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/gcd/Constt
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_1/y
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_1t
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_2/y
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_2t
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_3/y
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_3f
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/mul/y~
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/mul
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:2
stft/frame/concat/values_1r
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat/axisÜ
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P2 
stft/frame/concat_1/values_1/1ś
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1/values_1v
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_1/axisä
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1x
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
stft/frame/zeros_like
stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
stft/frame/ones_like/Shapez
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/ones_like/ConstŁ
stft/frame/ones_likeFill#stft/frame/ones_like/Shape:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
stft/frame/ones_likeâ
stft/frame/StridedSliceStridedSliceCast:y:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/StridedSliceŠ
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/Reshape_1v
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_1/startv
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_1/delta´
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/range_1
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/mul_1~
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_2/shape/1­
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_2/shape¤
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_2v
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_2/startv
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_2/deltaŽ
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
:2
stft/frame/range_2~
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_3/shape/0°
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_3/shape˘
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2
stft/frame/Reshape_3
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/add_1ç
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/GatherV2Ś
stft/frame/concat_2/values_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2/values_1v
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_2/axisä
stft/frame/concat_2ConcatV2stft/frame/split:output:0%stft/frame/concat_2/values_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2Ś
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_4x
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
stft/hann_window/periodic
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2
stft/hann_window/Cast|
stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/FloorMod/yĽ
stft/hann_window/FloorModFloorModstft/frame_length:output:0$stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/FloorModr
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub/x
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2
stft/hann_window/sub
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2
stft/hann_window/mul
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2
stft/hann_window/addv
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub_1/y
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/sub_1
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
stft/hann_window/Cast_1~
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/hann_window/range/start~
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/range/deltaŔ
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:2
stft/hann_window/range
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2
stft/hann_window/Cast_2y
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-DTű!@2
stft/hann_window/Const
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_1
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2
stft/hann_window/truedivw
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2
stft/hann_window/Cos}
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/mul_2/x
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_2}
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/sub_2/x
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2
stft/hann_window/sub_2
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

stft/mult
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:2
stft/rfft/packed
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   2
stft/rfft/Pad/paddings
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/rfft/Padw
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2
stft/rfft/fft_length
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*
Tcomplex0*
Treal0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	stft/rfftk
Abs
ComplexAbsstft/rfft:output:0*
T0*

Tout0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Abs
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :}2+
)linear_to_mel_weight_matrix/sample_rate/xž
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'linear_to_mel_weight_matrix/sample_rateĽ
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     @_@2.
,linear_to_mel_weight_matrix/lower_edge_hertzĽ
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     L˝@2.
,linear_to_mel_weight_matrix/upper_edge_hertz
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2#
!linear_to_mel_weight_matrix/Const
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @2'
%linear_to_mel_weight_matrix/truediv/yÓ
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2%
#linear_to_mel_weight_matrix/truediv
(linear_to_mel_weight_matrix/linspace/numConst*
_output_shapes
: *
dtype0*
value
B :2*
(linear_to_mel_weight_matrix/linspace/numÁ
)linear_to_mel_weight_matrix/linspace/CastCast1linear_to_mel_weight_matrix/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/linspace/CastÁ
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_1
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2,
*linear_to_mel_weight_matrix/linspace/Shape
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace/Shape_1˙
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastArgsů
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace/BroadcastToú
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1Ź
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/ExpandDims/dim
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/ExpandDims°
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace/ExpandDims_1Ś
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_2Ś
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_3ž
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8linear_to_mel_weight_matrix/linspace/strided_slice/stackÂ
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Â
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Â
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2linear_to_mel_weight_matrix/linspace/strided_slice
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2,
*linear_to_mel_weight_matrix/linspace/add/yđ
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/add´
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z29
7linear_to_mel_weight_matrix/linspace/SelectV2/condition¤
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/linspace/SelectV2/tľ
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace/SelectV2
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/linspace/sub/yŕ
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/sub˘
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 20
.linear_to_mel_weight_matrix/linspace/Maximum/yď
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace/Maximum
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace/sub_1/yć
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace/sub_1Ś
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :22
0linear_to_mel_weight_matrix/linspace/Maximum_1/y÷
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace/Maximum_1ú
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/sub_2Ć
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_2í
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/truedivŹ
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/GreaterEqual/y
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace/GreaterEqualą
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eś
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/SelectV2_1Ś
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/startŚ
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/deltaÔ
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/range/Castľ
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/rangeĚ
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:˙2-
+linear_to_mel_weight_matrix/linspace/Cast_3Ş
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2linear_to_mel_weight_matrix/linspace/range_1/startŞ
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace/range_1/delta¸
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/range_1ő
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/Equal¨
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eą
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_2ř
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:˙2.
,linear_to_mel_weight_matrix/linspace/Reshapeę
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:˙2*
(linear_to_mel_weight_matrix/linspace/mulď
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/add_1ó
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:2-
+linear_to_mel_weight_matrix/linspace/concatŹ
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 21
/linear_to_mel_weight_matrix/linspace/zeros_likeŠ
/linear_to_mel_weight_matrix/linspace/SelectV2_3SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:0-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/Shape_2:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_3ž
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:08linear_to_mel_weight_matrix/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes	
:2,
*linear_to_mel_weight_matrix/linspace/SliceŹ
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/linear_to_mel_weight_matrix/strided_slice/stack°
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/strided_slice/stack_1°
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/strided_slice/stack_2
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask2+
)linear_to_mel_weight_matrix/strided_sliceą
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@24
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/y
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:22
0linear_to_mel_weight_matrix/hertz_to_mel/truedivŠ
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?20
.linear_to_mel_weight_matrix/hertz_to_mel/add/xú
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/addť
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/LogŠ
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @20
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xô
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/mul
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/ExpandDims/dimď
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	2(
&linear_to_mel_weight_matrix/ExpandDimsľ
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_1/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/addź
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulľ
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_2/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/addź
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/mul
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :B2,
*linear_to_mel_weight_matrix/linspace_1/numÇ
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace_1/CastÇ
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_1
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace_1/ShapeŁ
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/linspace_1/Shape_1
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgs
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace_1/BroadcastTo
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1°
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/ExpandDims´
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_2Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_3Â
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackĆ
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Ć
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Î
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4linear_to_mel_weight_matrix/linspace_1/strided_slice
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2.
,linear_to_mel_weight_matrix/linspace_1/add/yř
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/add¸
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z2;
9linear_to_mel_weight_matrix/linspace_1/SelectV2/condition¨
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tż
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace_1/SelectV2
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace_1/sub/yč
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/subŚ
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 22
0linear_to_mel_weight_matrix/linspace_1/Maximum/y÷
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace_1/Maximum˘
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/linspace_1/sub_1/yî
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace_1/sub_1Ş
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/y˙
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace_1/Maximum_1
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/sub_2Ě
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_2ő
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/truediv°
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: 25
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualľ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eŔ
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1Ş
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/startŞ
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/deltaÚ
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/range/Castž
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/rangeŃ
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:@2/
-linear_to_mel_weight_matrix/linspace_1/Cast_3Ž
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4linear_to_mel_weight_matrix/linspace_1/range_1/startŽ
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaÂ
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/range_1ý
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/EqualŹ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eť
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2˙
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:@20
.linear_to_mel_weight_matrix/linspace_1/Reshapeń
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:@2,
*linear_to_mel_weight_matrix/linspace_1/mulö
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/add_1ţ
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:B2/
-linear_to_mel_weight_matrix/linspace_1/concat°
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/linspace_1/zeros_likeł
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:0/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3Ç
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:B2.
,linear_to_mel_weight_matrix/linspace_1/Slice˘
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/frame_length
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/frame/frame_step
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2(
&linear_to_mel_weight_matrix/frame/axis
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:B2)
'linear_to_mel_weight_matrix/frame/Shape
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/frame/Size/Const
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2(
&linear_to_mel_weight_matrix/frame/SizeŁ
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/frame/Size_1/Const
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(linear_to_mel_weight_matrix/frame/Size_1
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :B2)
'linear_to_mel_weight_matrix/frame/sub/xá
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/subç
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/frame/floordiv
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'linear_to_mel_weight_matrix/frame/add/xÚ
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/add
+linear_to_mel_weight_matrix/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2-
+linear_to_mel_weight_matrix/frame/Maximum/xă
)linear_to_mel_weight_matrix/frame/MaximumMaximum4linear_to_mel_weight_matrix/frame/Maximum/x:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/frame/Maximum
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/frame/gcd/Const˘
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_1/yű
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_1˘
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_2/yů
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_2Š
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_0°
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:B23
1linear_to_mel_weight_matrix/frame/concat/values_1Š
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_2 
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/concat/axisú
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(linear_to_mel_weight_matrix/frame/concat­
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_0ť
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"B      25
3linear_to_mel_weight_matrix/frame/concat_1/values_1­
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_2¤
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_1/axis
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_1´
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:B25
3linear_to_mel_weight_matrix/frame/zeros_like/tensorŚ
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2.
,linear_to_mel_weight_matrix/frame/zeros_like°
1linear_to_mel_weight_matrix/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/frame/ones_like/Shape¨
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/frame/ones_like/Const˙
+linear_to_mel_weight_matrix/frame/ones_likeFill:linear_to_mel_weight_matrix/frame/ones_like/Shape:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2-
+linear_to_mel_weight_matrix/frame/ones_likeů
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:B20
.linear_to_mel_weight_matrix/frame/StridedSliceř
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:B2+
)linear_to_mel_weight_matrix/frame/Reshape 
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/range/start 
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-linear_to_mel_weight_matrix/frame/range/delta
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:@2)
'linear_to_mel_weight_matrix/frame/rangeŢ
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:@2'
%linear_to_mel_weight_matrix/frame/mulŹ
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_1/shapeő
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_1¤
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/range_1/start¤
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/linear_to_mel_weight_matrix/frame/range_1/deltaĄ
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:2+
)linear_to_mel_weight_matrix/frame/range_1Ź
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_2/shapeţ
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:2-
+linear_to_mel_weight_matrix/frame/Reshape_2đ
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:@2)
'linear_to_mel_weight_matrix/frame/add_1¤
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/GatherV2/axisĎ
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:@2,
*linear_to_mel_weight_matrix/frame/GatherV2­
3linear_to_mel_weight_matrix/frame/concat_2/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_0
3linear_to_mel_weight_matrix/frame/concat_2/values_1Pack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/frame/concat_2/values_1­
3linear_to_mel_weight_matrix/frame/concat_2/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_2¤
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_2/axis
*linear_to_mel_weight_matrix/frame/concat_2ConcatV2<linear_to_mel_weight_matrix/frame/concat_2/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_2ř
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_3
#linear_to_mel_weight_matrix/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#linear_to_mel_weight_matrix/Const_1
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/split/split_dim
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:@:@:@*
	num_split2#
!linear_to_mel_weight_matrix/split§
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2+
)linear_to_mel_weight_matrix/Reshape/shapeŢ
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:@2%
#linear_to_mel_weight_matrix/ReshapeŤ
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_1/shapeä
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_1Ť
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_2/shapeä
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_2Ň
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	@2!
linear_to_mel_weight_matrix/subÔ
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_1Ď
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_1Ř
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	@2#
!linear_to_mel_weight_matrix/sub_2Ö
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_3Ń
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_2Ő
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/MinimumÔ
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/Maximum­
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2&
$linear_to_mel_weight_matrix/paddingsĂ
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	@2
linear_to_mel_weight_matrix{
matmulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
matmul_
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB 2ę-q=2
	Maximum/yu
MaximumMaximummatmul:product:0Maximum/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
MaximumW
add/yConst*
_output_shapes
: *
dtype0*
valueB 2üŠńŇMbP?2
add/yb
addAddV2Maximum:z:0add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
addL
LogLogadd:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Logj
frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :`2
frame/frame_lengthf
frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2
frame/frame_stepZ

frame/axisConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/axisQ
frame/ShapeShapeLog:y:0*
T0*
_output_shapes
:2
frame/ShapeZ

frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2

frame/Rankh
frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range/starth
frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range/delta
frame/rangeRangeframe/range/start:output:0frame/Rank:output:0frame/range/delta:output:0*
_output_shapes
:2
frame/range
frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/strided_slice/stack
frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_1
frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_2
frame/strided_sliceStridedSliceframe/range:output:0"frame/strided_slice/stack:output:0$frame/strided_slice/stack_1:output:0$frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
frame/strided_slice\
frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/sub/yi
	frame/subSubframe/Rank:output:0frame/sub/y:output:0*
T0*
_output_shapes
: 2
	frame/subo
frame/sub_1Subframe/sub:z:0frame/strided_slice:output:0*
T0*
_output_shapes
: 2
frame/sub_1b
frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/packed/1
frame/packedPackframe/strided_slice:output:0frame/packed/1:output:0frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
frame/packedp
frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/split/split_dim˝
frame/splitSplitVframe/Shape:output:0frame/packed:output:0frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
: ::*
	num_split2
frame/splitm
frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shapeq
frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shape_1
frame/ReshapeReshapeframe/split:output:1frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
frame/ReshapeZ

frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/Size^
frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Size_1w
frame/sub_2Subframe/Reshape:output:0frame/frame_length:output:0*
T0*
_output_shapes
: 2
frame/sub_2y
frame/floordivFloorDivframe/sub_2:z:0frame/frame_step:output:0*
T0*
_output_shapes
: 2
frame/floordiv\
frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
frame/add/xj
	frame/addAddV2frame/add/x:output:0frame/floordiv:z:0*
T0*
_output_shapes
: 2
	frame/addd
frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/Maximum/xs
frame/MaximumMaximumframe/Maximum/x:output:0frame/add:z:0*
T0*
_output_shapes
: 2
frame/Maximumd
frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/gcd/Constj
frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_1/y
frame/floordiv_1FloorDivframe/frame_length:output:0frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_1j
frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_2/y
frame/floordiv_2FloorDivframe/frame_step:output:0frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_2j
frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_3/y
frame/floordiv_3FloorDivframe/Reshape:output:0frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_3\
frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/mul/yj
	frame/mulMulframe/floordiv_3:z:0frame/mul/y:output:0*
T0*
_output_shapes
: 2
	frame/muls
frame/concat/values_1Packframe/mul:z:0*
N*
T0*
_output_shapes
:2
frame/concat/values_1h
frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat/axisž
frame/concatConcatV2frame/split:output:0frame/concat/values_1:output:0frame/split:output:2frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concatx
frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/concat_1/values_1/1˘
frame/concat_1/values_1Packframe/floordiv_3:z:0"frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1/values_1l
frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_1/axisĆ
frame/concat_1ConcatV2frame/split:output:0 frame/concat_1/values_1:output:0frame/split:output:2frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1n
frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/zeros_likex
frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
frame/ones_like/Shapep
frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/ones_like/Const
frame/ones_likeFillframe/ones_like/Shape:output:0frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
frame/ones_likeŐ
frame/StridedSliceStridedSliceLog:y:0frame/zeros_like:output:0frame/concat:output:0frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/StridedSlice˘
frame/Reshape_1Reshapeframe/StridedSlice:output:0frame/concat_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/Reshape_1l
frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_1/startl
frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_1/delta
frame/range_1Rangeframe/range_1/start:output:0frame/Maximum:z:0frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/range_1}
frame/mul_1Mulframe/range_1:output:0frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/mul_1t
frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_2/shape/1
frame/Reshape_2/shapePackframe/Maximum:z:0 frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
frame/Reshape_2/shape
frame/Reshape_2Reshapeframe/mul_1:z:0frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/Reshape_2l
frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_2/startl
frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_2/delta
frame/range_2Rangeframe/range_2/start:output:0frame/floordiv_1:z:0frame/range_2/delta:output:0*
_output_shapes
:`2
frame/range_2t
frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_3/shape/0
frame/Reshape_3/shapePack frame/Reshape_3/shape/0:output:0frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
frame/Reshape_3/shape
frame/Reshape_3Reshapeframe/range_2:output:0frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:`2
frame/Reshape_3
frame/add_1AddV2frame/Reshape_2:output:0frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`2
frame/add_1Ű
frame/GatherV2GatherV2frame/Reshape_1:output:0frame/add_1:z:0frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙`˙˙˙˙˙˙˙˙˙2
frame/GatherV2
frame/concat_2/values_1Packframe/Maximum:z:0frame/frame_length:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2/values_1l
frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_2/axisĆ
frame/concat_2ConcatV2frame/split:output:0 frame/concat_2/values_1:output:0frame/split:output:2frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2
frame/Reshape_4Reshapeframe/GatherV2:output:0frame/concat_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
frame/Reshape_4p
IdentityIdentityframe/Reshape_4:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2

Identity"
identityIdentity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
n
Ö
__inference___call___6458
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/AssertM
	map/ShapeShapesamples*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2ú
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2!
map/TensorArrayV2/element_shapeŔ
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2É
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeţ
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsamplesBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2#
!map/TensorArrayV2_1/element_shapeĆ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counterć
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
bodyR
map_while_body_6199*
condR
map_while_cond_6198*
output_shapes
: : : : : : 2
	map/whileÁ
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙`   @   26
4map/TensorArrayV2Stack/TensorListStack/element_shape
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStackm
ShapeShape/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
Shape
CastCast/map/TensorArrayV2Stack/TensorListStack:tensor:0*

DstT0*

SrcT0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@2
Castt
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ě
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
mulMulstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ě
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ě
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3
Reshape/shapePackmul:z:0strided_slice_2:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
Reshapeł
StatefulPartitionedCallStatefulPartitionedCallReshape:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2ě
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4q
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Reshape_1/shape/1i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2Ş
Reshape_1/shapePackstrided_slice_4:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshape StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	Reshape_1­
IdentityIdentityReshape_1:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ť
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate


map_while_body_5908$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorÍ
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeç
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItemč
map/while/PartitionedCallPartitionedCall4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_46702
map/while/PartitionedCallö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder"map/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*
_input_shapes
: : : : : : : 
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
: :

_output_shapes
: :

_output_shapes
: 
Ú
R
cond_false_3299
cond_placeholder
cond_identity_cast
cond_identityl
cond/IdentityIdentitycond_identity_cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

+
__inference__destroyer_6874
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 


map_while_body_6199$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorÍ
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeç
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItemč
map/while/PartitionedCallPartitionedCall4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_37272
map/while/PartitionedCallö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder"map/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*
_input_shapes
: : : : : : : 
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
: :

_output_shapes
: :

_output_shapes
: 
Ď
ü
__inference_<lambda>_7068
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identityö
PartitionedCallPartitionedCallunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*i
Tinb
`2^*
Tout
2*
_output_shapes
: *
_read_only_resource_inputsb
`^ 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_19312
PartitionedCallS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapesű
ř::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ĺ
;
$__inference__sample_to_features_3727
x
identityT
CastCastx*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Cast?
SizeSizeCast:y:0*
T0*
_output_shapes
: 2
SizeS
Less/yConst*
_output_shapes
: *
dtype0*
value
B :}2
Less/yU
LessLessSize:output:0Less/y:output:0*
T0*
_output_shapes
: 2
Lessź
condStatelessIfLess:z:0Size:output:0Cast:y:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *"
else_branchR
cond_false_3299*"
output_shapes
:˙˙˙˙˙˙˙˙˙*!
then_branchR
cond_true_32982
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identitym
Cast_1Castcond/Identity:output:0*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Cast_1i
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/frame_lengthe
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B : 2
stft/frame_stepe
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/fft_lengthm
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
stft/frame/axis^
stft/frame/ShapeShape
Cast_1:y:0*
T0*
_output_shapes
:2
stft/frame/Shaped
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Rankr
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range/startr
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range/deltaĽ
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:2
stft/frame/range
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
stft/frame/strided_slice/stack
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 stft/frame/strided_slice/stack_1
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 stft/frame/strided_slice/stack_2¤
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slicef
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub/y}
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_1l
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/packed/1ł
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packedz
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/split/split_dimÔ
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*
T0*

Tlen0*"
_output_shapes
: :: *
	num_split2
stft/frame/splitw
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape{
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape_1
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
stft/frame/Reshaped
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Sizeh
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Size_1
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_2
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: 2
stft/frame/floordivf
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/add/x~
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft/frame/addn
stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Maximum/x
stft/frame/MaximumMaximumstft/frame/Maximum/x:output:0stft/frame/add:z:0*
T0*
_output_shapes
: 2
stft/frame/Maximumn
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/gcd/Constt
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_1/y
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_1t
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_2/y
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_2t
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_3/y
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_3f
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/mul/y~
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/mul
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:2
stft/frame/concat/values_1r
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat/axisÜ
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P2 
stft/frame/concat_1/values_1/1ś
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1/values_1v
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_1/axisä
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1x
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
stft/frame/zeros_like
stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
stft/frame/ones_like/Shapez
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/ones_like/ConstŁ
stft/frame/ones_likeFill#stft/frame/ones_like/Shape:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
stft/frame/ones_likeä
stft/frame/StridedSliceStridedSlice
Cast_1:y:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/StridedSliceŠ
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/Reshape_1v
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_1/startv
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_1/delta´
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/range_1
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/mul_1~
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_2/shape/1­
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_2/shape¤
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_2v
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_2/startv
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_2/deltaŽ
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
:2
stft/frame/range_2~
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_3/shape/0°
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_3/shape˘
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2
stft/frame/Reshape_3
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/add_1ç
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/GatherV2Ś
stft/frame/concat_2/values_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2/values_1v
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_2/axisä
stft/frame/concat_2ConcatV2stft/frame/split:output:0%stft/frame/concat_2/values_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2Ś
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_4x
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
stft/hann_window/periodic
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2
stft/hann_window/Cast|
stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/FloorMod/yĽ
stft/hann_window/FloorModFloorModstft/frame_length:output:0$stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/FloorModr
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub/x
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2
stft/hann_window/sub
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2
stft/hann_window/mul
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2
stft/hann_window/addv
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub_1/y
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/sub_1
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
stft/hann_window/Cast_1~
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/hann_window/range/start~
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/range/deltaŔ
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:2
stft/hann_window/range
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2
stft/hann_window/Cast_2y
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-DTű!@2
stft/hann_window/Const
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_1
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2
stft/hann_window/truedivw
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2
stft/hann_window/Cos}
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/mul_2/x
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_2}
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/sub_2/x
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2
stft/hann_window/sub_2
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

stft/mult
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:2
stft/rfft/packed
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   2
stft/rfft/Pad/paddings
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/rfft/Padw
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2
stft/rfft/fft_length
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*
Tcomplex0*
Treal0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	stft/rfftk
Abs
ComplexAbsstft/rfft:output:0*
T0*

Tout0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Abs
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :}2+
)linear_to_mel_weight_matrix/sample_rate/xž
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'linear_to_mel_weight_matrix/sample_rateĽ
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     @_@2.
,linear_to_mel_weight_matrix/lower_edge_hertzĽ
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     L˝@2.
,linear_to_mel_weight_matrix/upper_edge_hertz
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2#
!linear_to_mel_weight_matrix/Const
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @2'
%linear_to_mel_weight_matrix/truediv/yÓ
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2%
#linear_to_mel_weight_matrix/truediv
(linear_to_mel_weight_matrix/linspace/numConst*
_output_shapes
: *
dtype0*
value
B :2*
(linear_to_mel_weight_matrix/linspace/numÁ
)linear_to_mel_weight_matrix/linspace/CastCast1linear_to_mel_weight_matrix/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/linspace/CastÁ
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_1
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2,
*linear_to_mel_weight_matrix/linspace/Shape
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace/Shape_1˙
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastArgsů
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace/BroadcastToú
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1Ź
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/ExpandDims/dim
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/ExpandDims°
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace/ExpandDims_1Ś
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_2Ś
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_3ž
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8linear_to_mel_weight_matrix/linspace/strided_slice/stackÂ
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Â
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Â
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2linear_to_mel_weight_matrix/linspace/strided_slice
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2,
*linear_to_mel_weight_matrix/linspace/add/yđ
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/add´
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z29
7linear_to_mel_weight_matrix/linspace/SelectV2/condition¤
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/linspace/SelectV2/tľ
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace/SelectV2
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/linspace/sub/yŕ
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/sub˘
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 20
.linear_to_mel_weight_matrix/linspace/Maximum/yď
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace/Maximum
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace/sub_1/yć
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace/sub_1Ś
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :22
0linear_to_mel_weight_matrix/linspace/Maximum_1/y÷
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace/Maximum_1ú
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/sub_2Ć
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_2í
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/truedivŹ
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/GreaterEqual/y
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace/GreaterEqualą
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eś
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/SelectV2_1Ś
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/startŚ
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/deltaÔ
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/range/Castľ
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/rangeĚ
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:˙2-
+linear_to_mel_weight_matrix/linspace/Cast_3Ş
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2linear_to_mel_weight_matrix/linspace/range_1/startŞ
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace/range_1/delta¸
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/range_1ő
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/Equal¨
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eą
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_2ř
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:˙2.
,linear_to_mel_weight_matrix/linspace/Reshapeę
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:˙2*
(linear_to_mel_weight_matrix/linspace/mulď
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/add_1ó
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:2-
+linear_to_mel_weight_matrix/linspace/concatŹ
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 21
/linear_to_mel_weight_matrix/linspace/zeros_likeŠ
/linear_to_mel_weight_matrix/linspace/SelectV2_3SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:0-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/Shape_2:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_3ž
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:08linear_to_mel_weight_matrix/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes	
:2,
*linear_to_mel_weight_matrix/linspace/SliceŹ
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/linear_to_mel_weight_matrix/strided_slice/stack°
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/strided_slice/stack_1°
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/strided_slice/stack_2
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask2+
)linear_to_mel_weight_matrix/strided_sliceą
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@24
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/y
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:22
0linear_to_mel_weight_matrix/hertz_to_mel/truedivŠ
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?20
.linear_to_mel_weight_matrix/hertz_to_mel/add/xú
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/addť
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/LogŠ
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @20
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xô
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/mul
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/ExpandDims/dimď
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	2(
&linear_to_mel_weight_matrix/ExpandDimsľ
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_1/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/addź
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulľ
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_2/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/addź
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/mul
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :B2,
*linear_to_mel_weight_matrix/linspace_1/numÇ
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace_1/CastÇ
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_1
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace_1/ShapeŁ
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/linspace_1/Shape_1
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgs
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace_1/BroadcastTo
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1°
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/ExpandDims´
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_2Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_3Â
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackĆ
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Ć
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Î
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4linear_to_mel_weight_matrix/linspace_1/strided_slice
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2.
,linear_to_mel_weight_matrix/linspace_1/add/yř
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/add¸
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z2;
9linear_to_mel_weight_matrix/linspace_1/SelectV2/condition¨
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tż
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace_1/SelectV2
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace_1/sub/yč
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/subŚ
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 22
0linear_to_mel_weight_matrix/linspace_1/Maximum/y÷
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace_1/Maximum˘
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/linspace_1/sub_1/yî
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace_1/sub_1Ş
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/y˙
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace_1/Maximum_1
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/sub_2Ě
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_2ő
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/truediv°
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: 25
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualľ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eŔ
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1Ş
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/startŞ
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/deltaÚ
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/range/Castž
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/rangeŃ
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:@2/
-linear_to_mel_weight_matrix/linspace_1/Cast_3Ž
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4linear_to_mel_weight_matrix/linspace_1/range_1/startŽ
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaÂ
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/range_1ý
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/EqualŹ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eť
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2˙
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:@20
.linear_to_mel_weight_matrix/linspace_1/Reshapeń
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:@2,
*linear_to_mel_weight_matrix/linspace_1/mulö
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/add_1ţ
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:B2/
-linear_to_mel_weight_matrix/linspace_1/concat°
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/linspace_1/zeros_likeł
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:0/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3Ç
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:B2.
,linear_to_mel_weight_matrix/linspace_1/Slice˘
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/frame_length
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/frame/frame_step
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2(
&linear_to_mel_weight_matrix/frame/axis
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:B2)
'linear_to_mel_weight_matrix/frame/Shape
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/frame/Size/Const
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2(
&linear_to_mel_weight_matrix/frame/SizeŁ
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/frame/Size_1/Const
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(linear_to_mel_weight_matrix/frame/Size_1
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :B2)
'linear_to_mel_weight_matrix/frame/sub/xá
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/subç
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/frame/floordiv
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'linear_to_mel_weight_matrix/frame/add/xÚ
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/add
+linear_to_mel_weight_matrix/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2-
+linear_to_mel_weight_matrix/frame/Maximum/xă
)linear_to_mel_weight_matrix/frame/MaximumMaximum4linear_to_mel_weight_matrix/frame/Maximum/x:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/frame/Maximum
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/frame/gcd/Const˘
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_1/yű
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_1˘
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_2/yů
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_2Š
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_0°
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:B23
1linear_to_mel_weight_matrix/frame/concat/values_1Š
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_2 
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/concat/axisú
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(linear_to_mel_weight_matrix/frame/concat­
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_0ť
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"B      25
3linear_to_mel_weight_matrix/frame/concat_1/values_1­
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_2¤
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_1/axis
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_1´
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:B25
3linear_to_mel_weight_matrix/frame/zeros_like/tensorŚ
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2.
,linear_to_mel_weight_matrix/frame/zeros_like°
1linear_to_mel_weight_matrix/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/frame/ones_like/Shape¨
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/frame/ones_like/Const˙
+linear_to_mel_weight_matrix/frame/ones_likeFill:linear_to_mel_weight_matrix/frame/ones_like/Shape:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2-
+linear_to_mel_weight_matrix/frame/ones_likeů
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:B20
.linear_to_mel_weight_matrix/frame/StridedSliceř
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:B2+
)linear_to_mel_weight_matrix/frame/Reshape 
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/range/start 
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-linear_to_mel_weight_matrix/frame/range/delta
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:@2)
'linear_to_mel_weight_matrix/frame/rangeŢ
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:@2'
%linear_to_mel_weight_matrix/frame/mulŹ
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_1/shapeő
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_1¤
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/range_1/start¤
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/linear_to_mel_weight_matrix/frame/range_1/deltaĄ
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:2+
)linear_to_mel_weight_matrix/frame/range_1Ź
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_2/shapeţ
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:2-
+linear_to_mel_weight_matrix/frame/Reshape_2đ
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:@2)
'linear_to_mel_weight_matrix/frame/add_1¤
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/GatherV2/axisĎ
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:@2,
*linear_to_mel_weight_matrix/frame/GatherV2­
3linear_to_mel_weight_matrix/frame/concat_2/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_0
3linear_to_mel_weight_matrix/frame/concat_2/values_1Pack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/frame/concat_2/values_1­
3linear_to_mel_weight_matrix/frame/concat_2/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_2¤
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_2/axis
*linear_to_mel_weight_matrix/frame/concat_2ConcatV2<linear_to_mel_weight_matrix/frame/concat_2/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_2ř
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_3
#linear_to_mel_weight_matrix/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#linear_to_mel_weight_matrix/Const_1
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/split/split_dim
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:@:@:@*
	num_split2#
!linear_to_mel_weight_matrix/split§
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2+
)linear_to_mel_weight_matrix/Reshape/shapeŢ
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:@2%
#linear_to_mel_weight_matrix/ReshapeŤ
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_1/shapeä
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_1Ť
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_2/shapeä
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_2Ň
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	@2!
linear_to_mel_weight_matrix/subÔ
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_1Ď
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_1Ř
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	@2#
!linear_to_mel_weight_matrix/sub_2Ö
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_3Ń
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_2Ő
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/MinimumÔ
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/Maximum­
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2&
$linear_to_mel_weight_matrix/paddingsĂ
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	@2
linear_to_mel_weight_matrix{
matmulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
matmul_
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB 2ę-q=2
	Maximum/yu
MaximumMaximummatmul:product:0Maximum/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
MaximumW
add/yConst*
_output_shapes
: *
dtype0*
valueB 2üŠńŇMbP?2
add/yb
addAddV2Maximum:z:0add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
addL
LogLogadd:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Logj
frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :`2
frame/frame_lengthf
frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2
frame/frame_stepZ

frame/axisConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/axisQ
frame/ShapeShapeLog:y:0*
T0*
_output_shapes
:2
frame/ShapeZ

frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2

frame/Rankh
frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range/starth
frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range/delta
frame/rangeRangeframe/range/start:output:0frame/Rank:output:0frame/range/delta:output:0*
_output_shapes
:2
frame/range
frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/strided_slice/stack
frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_1
frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_2
frame/strided_sliceStridedSliceframe/range:output:0"frame/strided_slice/stack:output:0$frame/strided_slice/stack_1:output:0$frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
frame/strided_slice\
frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/sub/yi
	frame/subSubframe/Rank:output:0frame/sub/y:output:0*
T0*
_output_shapes
: 2
	frame/subo
frame/sub_1Subframe/sub:z:0frame/strided_slice:output:0*
T0*
_output_shapes
: 2
frame/sub_1b
frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/packed/1
frame/packedPackframe/strided_slice:output:0frame/packed/1:output:0frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
frame/packedp
frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/split/split_dim˝
frame/splitSplitVframe/Shape:output:0frame/packed:output:0frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
: ::*
	num_split2
frame/splitm
frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shapeq
frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shape_1
frame/ReshapeReshapeframe/split:output:1frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
frame/ReshapeZ

frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/Size^
frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Size_1w
frame/sub_2Subframe/Reshape:output:0frame/frame_length:output:0*
T0*
_output_shapes
: 2
frame/sub_2y
frame/floordivFloorDivframe/sub_2:z:0frame/frame_step:output:0*
T0*
_output_shapes
: 2
frame/floordiv\
frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
frame/add/xj
	frame/addAddV2frame/add/x:output:0frame/floordiv:z:0*
T0*
_output_shapes
: 2
	frame/addd
frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/Maximum/xs
frame/MaximumMaximumframe/Maximum/x:output:0frame/add:z:0*
T0*
_output_shapes
: 2
frame/Maximumd
frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/gcd/Constj
frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_1/y
frame/floordiv_1FloorDivframe/frame_length:output:0frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_1j
frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_2/y
frame/floordiv_2FloorDivframe/frame_step:output:0frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_2j
frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_3/y
frame/floordiv_3FloorDivframe/Reshape:output:0frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_3\
frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/mul/yj
	frame/mulMulframe/floordiv_3:z:0frame/mul/y:output:0*
T0*
_output_shapes
: 2
	frame/muls
frame/concat/values_1Packframe/mul:z:0*
N*
T0*
_output_shapes
:2
frame/concat/values_1h
frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat/axisž
frame/concatConcatV2frame/split:output:0frame/concat/values_1:output:0frame/split:output:2frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concatx
frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/concat_1/values_1/1˘
frame/concat_1/values_1Packframe/floordiv_3:z:0"frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1/values_1l
frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_1/axisĆ
frame/concat_1ConcatV2frame/split:output:0 frame/concat_1/values_1:output:0frame/split:output:2frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1n
frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/zeros_likex
frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
frame/ones_like/Shapep
frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/ones_like/Const
frame/ones_likeFillframe/ones_like/Shape:output:0frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
frame/ones_likeŐ
frame/StridedSliceStridedSliceLog:y:0frame/zeros_like:output:0frame/concat:output:0frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/StridedSlice˘
frame/Reshape_1Reshapeframe/StridedSlice:output:0frame/concat_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/Reshape_1l
frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_1/startl
frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_1/delta
frame/range_1Rangeframe/range_1/start:output:0frame/Maximum:z:0frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/range_1}
frame/mul_1Mulframe/range_1:output:0frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/mul_1t
frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_2/shape/1
frame/Reshape_2/shapePackframe/Maximum:z:0 frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
frame/Reshape_2/shape
frame/Reshape_2Reshapeframe/mul_1:z:0frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/Reshape_2l
frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_2/startl
frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_2/delta
frame/range_2Rangeframe/range_2/start:output:0frame/floordiv_1:z:0frame/range_2/delta:output:0*
_output_shapes
:`2
frame/range_2t
frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_3/shape/0
frame/Reshape_3/shapePack frame/Reshape_3/shape/0:output:0frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
frame/Reshape_3/shape
frame/Reshape_3Reshapeframe/range_2:output:0frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:`2
frame/Reshape_3
frame/add_1AddV2frame/Reshape_2:output:0frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`2
frame/add_1Ű
frame/GatherV2GatherV2frame/Reshape_1:output:0frame/add_1:z:0frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙`˙˙˙˙˙˙˙˙˙2
frame/GatherV2
frame/concat_2/values_1Packframe/Maximum:z:0frame/frame_length:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2/values_1l
frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_2/axisĆ
frame/concat_2ConcatV2frame/split:output:0 frame/concat_2/values_1:output:0frame/split:output:2frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2
frame/Reshape_4Reshapeframe/GatherV2:output:0frame/concat_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
frame/Reshape_4p
IdentityIdentityframe/Reshape_4:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2

Identity"
identityIdentity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex

L
cond_true_4241
cond_sub_size
cond_pad_truediv
cond_identity[

cond/sub/xConst*
_output_shapes
: *
dtype0*
value
B :}2

cond/sub/x`
cond/subSubcond/sub/x:output:0cond_sub_size*
T0*
_output_shapes
: 2

cond/subp
cond/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Pad/paddings/0/0
cond/Pad/paddings/0Packcond/Pad/paddings/0/0:output:0cond/sub:z:0*
N*
T0*
_output_shapes
:2
cond/Pad/paddings/0~
cond/Pad/paddingsPackcond/Pad/paddings/0:output:0*
N*
T0*
_output_shapes

:2
cond/Pad/paddingsw
cond/PadPadcond_pad_truedivcond/Pad/paddings:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/Padk
cond/IdentityIdentitycond/Pad:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
9
Ö
__inference___call___5664
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/Assert§
PartitionedCallPartitionedCallsamples*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_46702
PartitionedCallV
ShapeShapePartitionedCall:output:0*
T0*
_output_shapes
:2
Shapes
CastCastPartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
CastŤ
StatefulPartitionedCallStatefulPartitionedCallCast:y:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallŽ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate

I
cond_true_3298
cond_sub_size
cond_pad_cast
cond_identity[

cond/sub/xConst*
_output_shapes
: *
dtype0*
value
B :}2

cond/sub/x`
cond/subSubcond/sub/x:output:0cond_sub_size*
T0*
_output_shapes
: 2

cond/subp
cond/Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Pad/paddings/0/0
cond/Pad/paddings/0Packcond/Pad/paddings/0/0:output:0cond/sub:z:0*
N*
T0*
_output_shapes
:2
cond/Pad/paddings/0~
cond/Pad/paddingsPackcond/Pad/paddings/0:output:0*
N*
T0*
_output_shapes

:2
cond/Pad/paddingst
cond/PadPadcond_pad_castcond/Pad/paddings:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/Padk
cond/IdentityIdentitycond/Pad:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*$
_input_shapes
: :˙˙˙˙˙˙˙˙˙: 

_output_shapes
: :)%
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 

map_while_cond_6198$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_6198___redundant_placeholder0
map_while_identity

map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*!
_input_shapes
: : : : : :: 
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
: :

_output_shapes
: :

_output_shapes
:
çž
6
__inference__traced_save_7377
file_prefix+
'savev2_save_counter_read_readvariableop	:
6savev2_network_layer1_conv_weights_read_readvariableopA
=savev2_network_layer1_conv_batchnorm_beta_read_readvariableopH
Dsavev2_network_layer1_conv_batchnorm_moving_mean_read_readvariableopL
Hsavev2_network_layer1_conv_batchnorm_moving_variance_read_readvariableopG
Csavev2_network_layer2_sepconv_depthwise_weights_read_readvariableopD
@savev2_network_layer2_sepconv_batchnorm_beta_read_readvariableopK
Gsavev2_network_layer2_sepconv_batchnorm_moving_mean_read_readvariableopO
Ksavev2_network_layer2_sepconv_batchnorm_moving_variance_read_readvariableop:
6savev2_network_layer3_conv_weights_read_readvariableopA
=savev2_network_layer3_conv_batchnorm_beta_read_readvariableopH
Dsavev2_network_layer3_conv_batchnorm_moving_mean_read_readvariableopL
Hsavev2_network_layer3_conv_batchnorm_moving_variance_read_readvariableopG
Csavev2_network_layer4_sepconv_depthwise_weights_read_readvariableopD
@savev2_network_layer4_sepconv_batchnorm_beta_read_readvariableopK
Gsavev2_network_layer4_sepconv_batchnorm_moving_mean_read_readvariableopO
Ksavev2_network_layer4_sepconv_batchnorm_moving_variance_read_readvariableop:
6savev2_network_layer5_conv_weights_read_readvariableopA
=savev2_network_layer5_conv_batchnorm_beta_read_readvariableopH
Dsavev2_network_layer5_conv_batchnorm_moving_mean_read_readvariableopL
Hsavev2_network_layer5_conv_batchnorm_moving_variance_read_readvariableopG
Csavev2_network_layer6_sepconv_depthwise_weights_read_readvariableopD
@savev2_network_layer6_sepconv_batchnorm_beta_read_readvariableopK
Gsavev2_network_layer6_sepconv_batchnorm_moving_mean_read_readvariableopO
Ksavev2_network_layer6_sepconv_batchnorm_moving_variance_read_readvariableop:
6savev2_network_layer7_conv_weights_read_readvariableopA
=savev2_network_layer7_conv_batchnorm_beta_read_readvariableopH
Dsavev2_network_layer7_conv_batchnorm_moving_mean_read_readvariableopL
Hsavev2_network_layer7_conv_batchnorm_moving_variance_read_readvariableopG
Csavev2_network_layer8_sepconv_depthwise_weights_read_readvariableopD
@savev2_network_layer8_sepconv_batchnorm_beta_read_readvariableopK
Gsavev2_network_layer8_sepconv_batchnorm_moving_mean_read_readvariableopO
Ksavev2_network_layer8_sepconv_batchnorm_moving_variance_read_readvariableop:
6savev2_network_layer9_conv_weights_read_readvariableopA
=savev2_network_layer9_conv_batchnorm_beta_read_readvariableopH
Dsavev2_network_layer9_conv_batchnorm_moving_mean_read_readvariableopL
Hsavev2_network_layer9_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer10_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer10_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer10_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer10_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer11_conv_weights_read_readvariableopB
>savev2_network_layer11_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer11_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer11_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer12_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer12_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer12_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer12_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer13_conv_weights_read_readvariableopB
>savev2_network_layer13_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer13_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer13_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer14_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer14_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer14_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer14_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer15_conv_weights_read_readvariableopB
>savev2_network_layer15_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer15_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer15_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer16_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer16_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer16_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer16_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer17_conv_weights_read_readvariableopB
>savev2_network_layer17_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer17_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer17_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer18_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer18_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer18_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer18_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer19_conv_weights_read_readvariableopB
>savev2_network_layer19_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer19_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer19_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer20_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer20_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer20_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer20_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer21_conv_weights_read_readvariableopB
>savev2_network_layer21_conv_batchnorm_beta_read_readvariableopI
Esavev2_network_layer21_conv_batchnorm_moving_mean_read_readvariableopM
Isavev2_network_layer21_conv_batchnorm_moving_variance_read_readvariableopH
Dsavev2_network_layer22_sepconv_depthwise_weights_read_readvariableopE
Asavev2_network_layer22_sepconv_batchnorm_beta_read_readvariableopL
Hsavev2_network_layer22_sepconv_batchnorm_moving_mean_read_readvariableopP
Lsavev2_network_layer22_sepconv_batchnorm_moving_variance_read_readvariableop;
7savev2_network_layer23_conv_weights_read_readvariableop:
6savev2_network_layer23_conv_biases_read_readvariableop9
5savev2_network_layer25_fc_weights_read_readvariableop8
4savev2_network_layer25_fc_biases_read_readvariableop9
5savev2_network_layer28_fc_weights_read_readvariableop8
4savev2_network_layer28_fc_biases_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d9c7ac58416546fca05e8fa971569b0c/part2	
Const_1
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÎ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*ŕ
valueÖBÓ`B'save_counter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesË
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:`*
dtype0*Ő
valueËBČ`B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÍ4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_save_counter_read_readvariableop6savev2_network_layer1_conv_weights_read_readvariableop=savev2_network_layer1_conv_batchnorm_beta_read_readvariableopDsavev2_network_layer1_conv_batchnorm_moving_mean_read_readvariableopHsavev2_network_layer1_conv_batchnorm_moving_variance_read_readvariableopCsavev2_network_layer2_sepconv_depthwise_weights_read_readvariableop@savev2_network_layer2_sepconv_batchnorm_beta_read_readvariableopGsavev2_network_layer2_sepconv_batchnorm_moving_mean_read_readvariableopKsavev2_network_layer2_sepconv_batchnorm_moving_variance_read_readvariableop6savev2_network_layer3_conv_weights_read_readvariableop=savev2_network_layer3_conv_batchnorm_beta_read_readvariableopDsavev2_network_layer3_conv_batchnorm_moving_mean_read_readvariableopHsavev2_network_layer3_conv_batchnorm_moving_variance_read_readvariableopCsavev2_network_layer4_sepconv_depthwise_weights_read_readvariableop@savev2_network_layer4_sepconv_batchnorm_beta_read_readvariableopGsavev2_network_layer4_sepconv_batchnorm_moving_mean_read_readvariableopKsavev2_network_layer4_sepconv_batchnorm_moving_variance_read_readvariableop6savev2_network_layer5_conv_weights_read_readvariableop=savev2_network_layer5_conv_batchnorm_beta_read_readvariableopDsavev2_network_layer5_conv_batchnorm_moving_mean_read_readvariableopHsavev2_network_layer5_conv_batchnorm_moving_variance_read_readvariableopCsavev2_network_layer6_sepconv_depthwise_weights_read_readvariableop@savev2_network_layer6_sepconv_batchnorm_beta_read_readvariableopGsavev2_network_layer6_sepconv_batchnorm_moving_mean_read_readvariableopKsavev2_network_layer6_sepconv_batchnorm_moving_variance_read_readvariableop6savev2_network_layer7_conv_weights_read_readvariableop=savev2_network_layer7_conv_batchnorm_beta_read_readvariableopDsavev2_network_layer7_conv_batchnorm_moving_mean_read_readvariableopHsavev2_network_layer7_conv_batchnorm_moving_variance_read_readvariableopCsavev2_network_layer8_sepconv_depthwise_weights_read_readvariableop@savev2_network_layer8_sepconv_batchnorm_beta_read_readvariableopGsavev2_network_layer8_sepconv_batchnorm_moving_mean_read_readvariableopKsavev2_network_layer8_sepconv_batchnorm_moving_variance_read_readvariableop6savev2_network_layer9_conv_weights_read_readvariableop=savev2_network_layer9_conv_batchnorm_beta_read_readvariableopDsavev2_network_layer9_conv_batchnorm_moving_mean_read_readvariableopHsavev2_network_layer9_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer10_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer10_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer10_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer10_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer11_conv_weights_read_readvariableop>savev2_network_layer11_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer11_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer11_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer12_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer12_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer12_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer12_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer13_conv_weights_read_readvariableop>savev2_network_layer13_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer13_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer13_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer14_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer14_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer14_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer14_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer15_conv_weights_read_readvariableop>savev2_network_layer15_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer15_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer15_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer16_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer16_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer16_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer16_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer17_conv_weights_read_readvariableop>savev2_network_layer17_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer17_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer17_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer18_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer18_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer18_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer18_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer19_conv_weights_read_readvariableop>savev2_network_layer19_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer19_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer19_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer20_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer20_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer20_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer20_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer21_conv_weights_read_readvariableop>savev2_network_layer21_conv_batchnorm_beta_read_readvariableopEsavev2_network_layer21_conv_batchnorm_moving_mean_read_readvariableopIsavev2_network_layer21_conv_batchnorm_moving_variance_read_readvariableopDsavev2_network_layer22_sepconv_depthwise_weights_read_readvariableopAsavev2_network_layer22_sepconv_batchnorm_beta_read_readvariableopHsavev2_network_layer22_sepconv_batchnorm_moving_mean_read_readvariableopLsavev2_network_layer22_sepconv_batchnorm_moving_variance_read_readvariableop7savev2_network_layer23_conv_weights_read_readvariableop6savev2_network_layer23_conv_biases_read_readvariableop5savev2_network_layer25_fc_weights_read_readvariableop4savev2_network_layer25_fc_biases_read_readvariableop5savev2_network_layer28_fc_weights_read_readvariableop4savev2_network_layer28_fc_biases_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *n
dtypesd
b2`	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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

identity_1Identity_1:output:0*Ä
_input_shapes˛
Ż: : : : : : : : : : : @:@:@:@:@:@:@:@:@::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
`::
`:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::-&)
'
_output_shapes
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::.**
(
_output_shapes
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::-.)
'
_output_shapes
::!/

_output_shapes	
::!0

_output_shapes	
::!1

_output_shapes	
::.2*
(
_output_shapes
::!3

_output_shapes	
::!4

_output_shapes	
::!5

_output_shapes	
::-6)
'
_output_shapes
::!7

_output_shapes	
::!8

_output_shapes	
::!9

_output_shapes	
::.:*
(
_output_shapes
::!;

_output_shapes	
::!<

_output_shapes	
::!=

_output_shapes	
::->)
'
_output_shapes
::!?

_output_shapes	
::!@

_output_shapes	
::!A

_output_shapes	
::.B*
(
_output_shapes
::!C

_output_shapes	
::!D

_output_shapes	
::!E

_output_shapes	
::-F)
'
_output_shapes
::!G

_output_shapes	
::!H

_output_shapes	
::!I

_output_shapes	
::.J*
(
_output_shapes
::!K

_output_shapes	
::!L

_output_shapes	
::!M

_output_shapes	
::-N)
'
_output_shapes
::!O

_output_shapes	
::!P

_output_shapes	
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::-V)
'
_output_shapes
::!W

_output_shapes	
::!X

_output_shapes	
::!Y

_output_shapes	
::.Z*
(
_output_shapes
::![

_output_shapes	
::&\"
 
_output_shapes
:
`:!]

_output_shapes	
::&^"
 
_output_shapes
:
`:!_

_output_shapes	
:`:`

_output_shapes
: 
9
Ö
__inference___call___5876
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/Assert§
PartitionedCallPartitionedCallsamples*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_37272
PartitionedCallV
ShapeShapePartitionedCall:output:0*
T0*
_output_shapes
:2
Shapes
CastCastPartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
CastŤ
StatefulPartitionedCallStatefulPartitionedCallCast:y:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallŽ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate
Ě¸
đ!
__inference_pruned_2330
inference_input
network_layer1_conv_weights&
"network_layer1_conv_batchnorm_beta-
)network_layer1_conv_batchnorm_moving_mean1
-network_layer1_conv_batchnorm_moving_variance,
(network_layer2_sepconv_depthwise_weights)
%network_layer2_sepconv_batchnorm_beta0
,network_layer2_sepconv_batchnorm_moving_mean4
0network_layer2_sepconv_batchnorm_moving_variance
network_layer3_conv_weights&
"network_layer3_conv_batchnorm_beta-
)network_layer3_conv_batchnorm_moving_mean1
-network_layer3_conv_batchnorm_moving_variance,
(network_layer4_sepconv_depthwise_weights)
%network_layer4_sepconv_batchnorm_beta0
,network_layer4_sepconv_batchnorm_moving_mean4
0network_layer4_sepconv_batchnorm_moving_variance
network_layer5_conv_weights&
"network_layer5_conv_batchnorm_beta-
)network_layer5_conv_batchnorm_moving_mean1
-network_layer5_conv_batchnorm_moving_variance,
(network_layer6_sepconv_depthwise_weights)
%network_layer6_sepconv_batchnorm_beta0
,network_layer6_sepconv_batchnorm_moving_mean4
0network_layer6_sepconv_batchnorm_moving_variance
network_layer7_conv_weights&
"network_layer7_conv_batchnorm_beta-
)network_layer7_conv_batchnorm_moving_mean1
-network_layer7_conv_batchnorm_moving_variance,
(network_layer8_sepconv_depthwise_weights)
%network_layer8_sepconv_batchnorm_beta0
,network_layer8_sepconv_batchnorm_moving_mean4
0network_layer8_sepconv_batchnorm_moving_variance
network_layer9_conv_weights&
"network_layer9_conv_batchnorm_beta-
)network_layer9_conv_batchnorm_moving_mean1
-network_layer9_conv_batchnorm_moving_variance-
)network_layer10_sepconv_depthwise_weights*
&network_layer10_sepconv_batchnorm_beta1
-network_layer10_sepconv_batchnorm_moving_mean5
1network_layer10_sepconv_batchnorm_moving_variance 
network_layer11_conv_weights'
#network_layer11_conv_batchnorm_beta.
*network_layer11_conv_batchnorm_moving_mean2
.network_layer11_conv_batchnorm_moving_variance-
)network_layer12_sepconv_depthwise_weights*
&network_layer12_sepconv_batchnorm_beta1
-network_layer12_sepconv_batchnorm_moving_mean5
1network_layer12_sepconv_batchnorm_moving_variance 
network_layer13_conv_weights'
#network_layer13_conv_batchnorm_beta.
*network_layer13_conv_batchnorm_moving_mean2
.network_layer13_conv_batchnorm_moving_variance-
)network_layer14_sepconv_depthwise_weights*
&network_layer14_sepconv_batchnorm_beta1
-network_layer14_sepconv_batchnorm_moving_mean5
1network_layer14_sepconv_batchnorm_moving_variance 
network_layer15_conv_weights'
#network_layer15_conv_batchnorm_beta.
*network_layer15_conv_batchnorm_moving_mean2
.network_layer15_conv_batchnorm_moving_variance-
)network_layer16_sepconv_depthwise_weights*
&network_layer16_sepconv_batchnorm_beta1
-network_layer16_sepconv_batchnorm_moving_mean5
1network_layer16_sepconv_batchnorm_moving_variance 
network_layer17_conv_weights'
#network_layer17_conv_batchnorm_beta.
*network_layer17_conv_batchnorm_moving_mean2
.network_layer17_conv_batchnorm_moving_variance-
)network_layer18_sepconv_depthwise_weights*
&network_layer18_sepconv_batchnorm_beta1
-network_layer18_sepconv_batchnorm_moving_mean5
1network_layer18_sepconv_batchnorm_moving_variance 
network_layer19_conv_weights'
#network_layer19_conv_batchnorm_beta.
*network_layer19_conv_batchnorm_moving_mean2
.network_layer19_conv_batchnorm_moving_variance-
)network_layer20_sepconv_depthwise_weights*
&network_layer20_sepconv_batchnorm_beta1
-network_layer20_sepconv_batchnorm_moving_mean5
1network_layer20_sepconv_batchnorm_moving_variance 
network_layer21_conv_weights'
#network_layer21_conv_batchnorm_beta.
*network_layer21_conv_batchnorm_moving_mean2
.network_layer21_conv_batchnorm_moving_variance-
)network_layer22_sepconv_depthwise_weights*
&network_layer22_sepconv_batchnorm_beta1
-network_layer22_sepconv_batchnorm_moving_mean5
1network_layer22_sepconv_batchnorm_moving_variance 
network_layer23_conv_weights
network_layer23_conv_biases
network_layer25_fc_weights
network_layer25_fc_biases
network_layer28_fc_weights
network_layer28_fc_biases$
 tower0_network_layer26_embeddingx
pre_tower/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pre_tower/split/split_dimb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinference_inputExpandDims/dim:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2

ExpandDimsŻ
pre_tower/splitSplit"pre_tower/split/split_dim:output:0ExpandDims:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@*
	num_split2
pre_tower/splitČ
0tower0/network/layer1/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer1_conv_weights*&
_output_shapes
: *
dtype022
0tower0/network/layer1/conv/Conv2D/ReadVariableOp
!tower0/network/layer1/conv/Conv2DConv2Dpre_tower/split:output:08tower0/network/layer1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0  *
paddingSAME*
strides
2#
!tower0/network/layer1/conv/Conv2DĽ
*tower0/network/layer1/conv/BatchNorm/ConstConst*
_output_shapes
: *
dtype0*
valueB *  ?2,
*tower0/network/layer1/conv/BatchNorm/ConstÉ
3tower0/network/layer1/conv/BatchNorm/ReadVariableOpReadVariableOp"network_layer1_conv_batchnorm_beta*
_output_shapes
: *
dtype025
3tower0/network/layer1/conv/BatchNorm/ReadVariableOpň
Dtower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp)network_layer1_conv_batchnorm_moving_mean*
_output_shapes
: *
dtype02F
Dtower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpú
Ftower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp-network_layer1_conv_batchnorm_moving_variance*
_output_shapes
: *
dtype02H
Ftower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1°
5tower0/network/layer1/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3*tower0/network/layer1/conv/Conv2D:output:03tower0/network/layer1/conv/BatchNorm/Const:output:0;tower0/network/layer1/conv/BatchNorm/ReadVariableOp:value:0Ltower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ntower0/network/layer1/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙0  : : : : :*
is_training( 27
5tower0/network/layer1/conv/BatchNorm/FusedBatchNormV3ż
tower0/network/layer1/conv/ReluRelu9tower0/network/layer1/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0  2!
tower0/network/layer1/conv/Reluá
6tower0/network/layer2/sepconv/depthwise/ReadVariableOpReadVariableOp(network_layer2_sepconv_depthwise_weights*&
_output_shapes
: *
dtype028
6tower0/network/layer2/sepconv/depthwise/ReadVariableOpź
'tower0/network/layer2/sepconv/depthwiseDepthwiseConv2dNative-tower0/network/layer1/conv/Relu:activations:0>tower0/network/layer2/sepconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0  *
paddingSAME*
strides
2)
'tower0/network/layer2/sepconv/depthwiseŤ
-tower0/network/layer2/sepconv/BatchNorm/ConstConst*
_output_shapes
: *
dtype0*
valueB *  ?2/
-tower0/network/layer2/sepconv/BatchNorm/ConstŇ
6tower0/network/layer2/sepconv/BatchNorm/ReadVariableOpReadVariableOp%network_layer2_sepconv_batchnorm_beta*
_output_shapes
: *
dtype028
6tower0/network/layer2/sepconv/BatchNorm/ReadVariableOpű
Gtower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp,network_layer2_sepconv_batchnorm_moving_mean*
_output_shapes
: *
dtype02I
Gtower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Itower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0network_layer2_sepconv_batchnorm_moving_variance*
_output_shapes
: *
dtype02K
Itower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Č
8tower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV30tower0/network/layer2/sepconv/depthwise:output:06tower0/network/layer2/sepconv/BatchNorm/Const:output:0>tower0/network/layer2/sepconv/BatchNorm/ReadVariableOp:value:0Otower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Qtower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙0  : : : : :*
is_training( 2:
8tower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3Č
"tower0/network/layer2/sepconv/ReluRelu<tower0/network/layer2/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0  2$
"tower0/network/layer2/sepconv/ReluČ
0tower0/network/layer3/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer3_conv_weights*&
_output_shapes
: @*
dtype022
0tower0/network/layer3/conv/Conv2D/ReadVariableOp
!tower0/network/layer3/conv/Conv2DConv2D0tower0/network/layer2/sepconv/Relu:activations:08tower0/network/layer3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0 @*
paddingSAME*
strides
2#
!tower0/network/layer3/conv/Conv2DĽ
*tower0/network/layer3/conv/BatchNorm/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*  ?2,
*tower0/network/layer3/conv/BatchNorm/ConstÉ
3tower0/network/layer3/conv/BatchNorm/ReadVariableOpReadVariableOp"network_layer3_conv_batchnorm_beta*
_output_shapes
:@*
dtype025
3tower0/network/layer3/conv/BatchNorm/ReadVariableOpň
Dtower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp)network_layer3_conv_batchnorm_moving_mean*
_output_shapes
:@*
dtype02F
Dtower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpú
Ftower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp-network_layer3_conv_batchnorm_moving_variance*
_output_shapes
:@*
dtype02H
Ftower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1°
5tower0/network/layer3/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3*tower0/network/layer3/conv/Conv2D:output:03tower0/network/layer3/conv/BatchNorm/Const:output:0;tower0/network/layer3/conv/BatchNorm/ReadVariableOp:value:0Ltower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ntower0/network/layer3/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙0 @:@:@:@:@:*
is_training( 27
5tower0/network/layer3/conv/BatchNorm/FusedBatchNormV3ż
tower0/network/layer3/conv/ReluRelu9tower0/network/layer3/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0 @2!
tower0/network/layer3/conv/Reluá
6tower0/network/layer4/sepconv/depthwise/ReadVariableOpReadVariableOp(network_layer4_sepconv_depthwise_weights*&
_output_shapes
:@*
dtype028
6tower0/network/layer4/sepconv/depthwise/ReadVariableOpź
'tower0/network/layer4/sepconv/depthwiseDepthwiseConv2dNative-tower0/network/layer3/conv/Relu:activations:0>tower0/network/layer4/sepconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
2)
'tower0/network/layer4/sepconv/depthwiseŤ
-tower0/network/layer4/sepconv/BatchNorm/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*  ?2/
-tower0/network/layer4/sepconv/BatchNorm/ConstŇ
6tower0/network/layer4/sepconv/BatchNorm/ReadVariableOpReadVariableOp%network_layer4_sepconv_batchnorm_beta*
_output_shapes
:@*
dtype028
6tower0/network/layer4/sepconv/BatchNorm/ReadVariableOpű
Gtower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp,network_layer4_sepconv_batchnorm_moving_mean*
_output_shapes
:@*
dtype02I
Gtower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Itower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0network_layer4_sepconv_batchnorm_moving_variance*
_output_shapes
:@*
dtype02K
Itower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Č
8tower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV30tower0/network/layer4/sepconv/depthwise:output:06tower0/network/layer4/sepconv/BatchNorm/Const:output:0>tower0/network/layer4/sepconv/BatchNorm/ReadVariableOp:value:0Otower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Qtower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙@:@:@:@:@:*
is_training( 2:
8tower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3Č
"tower0/network/layer4/sepconv/ReluRelu<tower0/network/layer4/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@2$
"tower0/network/layer4/sepconv/ReluÉ
0tower0/network/layer5/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer5_conv_weights*'
_output_shapes
:@*
dtype022
0tower0/network/layer5/conv/Conv2D/ReadVariableOp
!tower0/network/layer5/conv/Conv2DConv2D0tower0/network/layer4/sepconv/Relu:activations:08tower0/network/layer5/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2#
!tower0/network/layer5/conv/Conv2D§
*tower0/network/layer5/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2,
*tower0/network/layer5/conv/BatchNorm/ConstĘ
3tower0/network/layer5/conv/BatchNorm/ReadVariableOpReadVariableOp"network_layer5_conv_batchnorm_beta*
_output_shapes	
:*
dtype025
3tower0/network/layer5/conv/BatchNorm/ReadVariableOpó
Dtower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp)network_layer5_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02F
Dtower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpű
Ftower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp-network_layer5_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02H
Ftower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ľ
5tower0/network/layer5/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3*tower0/network/layer5/conv/Conv2D:output:03tower0/network/layer5/conv/BatchNorm/Const:output:0;tower0/network/layer5/conv/BatchNorm/ReadVariableOp:value:0Ltower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ntower0/network/layer5/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 27
5tower0/network/layer5/conv/BatchNorm/FusedBatchNormV3Ŕ
tower0/network/layer5/conv/ReluRelu9tower0/network/layer5/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
tower0/network/layer5/conv/Reluâ
6tower0/network/layer6/sepconv/depthwise/ReadVariableOpReadVariableOp(network_layer6_sepconv_depthwise_weights*'
_output_shapes
:*
dtype028
6tower0/network/layer6/sepconv/depthwise/ReadVariableOp˝
'tower0/network/layer6/sepconv/depthwiseDepthwiseConv2dNative-tower0/network/layer5/conv/Relu:activations:0>tower0/network/layer6/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2)
'tower0/network/layer6/sepconv/depthwise­
-tower0/network/layer6/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2/
-tower0/network/layer6/sepconv/BatchNorm/ConstÓ
6tower0/network/layer6/sepconv/BatchNorm/ReadVariableOpReadVariableOp%network_layer6_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype028
6tower0/network/layer6/sepconv/BatchNorm/ReadVariableOpü
Gtower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp,network_layer6_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02I
Gtower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Itower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0network_layer6_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02K
Itower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Í
8tower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV30tower0/network/layer6/sepconv/depthwise:output:06tower0/network/layer6/sepconv/BatchNorm/Const:output:0>tower0/network/layer6/sepconv/BatchNorm/ReadVariableOp:value:0Otower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Qtower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2:
8tower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3É
"tower0/network/layer6/sepconv/ReluRelu<tower0/network/layer6/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"tower0/network/layer6/sepconv/ReluĘ
0tower0/network/layer7/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer7_conv_weights*(
_output_shapes
:*
dtype022
0tower0/network/layer7/conv/Conv2D/ReadVariableOp
!tower0/network/layer7/conv/Conv2DConv2D0tower0/network/layer6/sepconv/Relu:activations:08tower0/network/layer7/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2#
!tower0/network/layer7/conv/Conv2D§
*tower0/network/layer7/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2,
*tower0/network/layer7/conv/BatchNorm/ConstĘ
3tower0/network/layer7/conv/BatchNorm/ReadVariableOpReadVariableOp"network_layer7_conv_batchnorm_beta*
_output_shapes	
:*
dtype025
3tower0/network/layer7/conv/BatchNorm/ReadVariableOpó
Dtower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp)network_layer7_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02F
Dtower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpű
Ftower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp-network_layer7_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02H
Ftower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ľ
5tower0/network/layer7/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3*tower0/network/layer7/conv/Conv2D:output:03tower0/network/layer7/conv/BatchNorm/Const:output:0;tower0/network/layer7/conv/BatchNorm/ReadVariableOp:value:0Ltower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ntower0/network/layer7/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 27
5tower0/network/layer7/conv/BatchNorm/FusedBatchNormV3Ŕ
tower0/network/layer7/conv/ReluRelu9tower0/network/layer7/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
tower0/network/layer7/conv/Reluâ
6tower0/network/layer8/sepconv/depthwise/ReadVariableOpReadVariableOp(network_layer8_sepconv_depthwise_weights*'
_output_shapes
:*
dtype028
6tower0/network/layer8/sepconv/depthwise/ReadVariableOp˝
'tower0/network/layer8/sepconv/depthwiseDepthwiseConv2dNative-tower0/network/layer7/conv/Relu:activations:0>tower0/network/layer8/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2)
'tower0/network/layer8/sepconv/depthwise­
-tower0/network/layer8/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2/
-tower0/network/layer8/sepconv/BatchNorm/ConstÓ
6tower0/network/layer8/sepconv/BatchNorm/ReadVariableOpReadVariableOp%network_layer8_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype028
6tower0/network/layer8/sepconv/BatchNorm/ReadVariableOpü
Gtower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp,network_layer8_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02I
Gtower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Itower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0network_layer8_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02K
Itower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Í
8tower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV30tower0/network/layer8/sepconv/depthwise:output:06tower0/network/layer8/sepconv/BatchNorm/Const:output:0>tower0/network/layer8/sepconv/BatchNorm/ReadVariableOp:value:0Otower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Qtower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2:
8tower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3É
"tower0/network/layer8/sepconv/ReluRelu<tower0/network/layer8/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"tower0/network/layer8/sepconv/ReluĘ
0tower0/network/layer9/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer9_conv_weights*(
_output_shapes
:*
dtype022
0tower0/network/layer9/conv/Conv2D/ReadVariableOp
!tower0/network/layer9/conv/Conv2DConv2D0tower0/network/layer8/sepconv/Relu:activations:08tower0/network/layer9/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2#
!tower0/network/layer9/conv/Conv2D§
*tower0/network/layer9/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2,
*tower0/network/layer9/conv/BatchNorm/ConstĘ
3tower0/network/layer9/conv/BatchNorm/ReadVariableOpReadVariableOp"network_layer9_conv_batchnorm_beta*
_output_shapes	
:*
dtype025
3tower0/network/layer9/conv/BatchNorm/ReadVariableOpó
Dtower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp)network_layer9_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02F
Dtower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpű
Ftower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp-network_layer9_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02H
Ftower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ľ
5tower0/network/layer9/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3*tower0/network/layer9/conv/Conv2D:output:03tower0/network/layer9/conv/BatchNorm/Const:output:0;tower0/network/layer9/conv/BatchNorm/ReadVariableOp:value:0Ltower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Ntower0/network/layer9/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 27
5tower0/network/layer9/conv/BatchNorm/FusedBatchNormV3Ŕ
tower0/network/layer9/conv/ReluRelu9tower0/network/layer9/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
tower0/network/layer9/conv/Reluĺ
7tower0/network/layer10/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer10_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer10/sepconv/depthwise/ReadVariableOpŔ
(tower0/network/layer10/sepconv/depthwiseDepthwiseConv2dNative-tower0/network/layer9/conv/Relu:activations:0?tower0/network/layer10/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer10/sepconv/depthwiseŻ
.tower0/network/layer10/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer10/sepconv/BatchNorm/ConstÖ
7tower0/network/layer10/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer10_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer10/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer10_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer10_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer10/sepconv/depthwise:output:07tower0/network/layer10/sepconv/BatchNorm/Const:output:0?tower0/network/layer10/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer10/sepconv/ReluRelu=tower0/network/layer10/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer10/sepconv/ReluÍ
1tower0/network/layer11/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer11_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer11/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer11/conv/Conv2DConv2D1tower0/network/layer10/sepconv/Relu:activations:09tower0/network/layer11/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer11/conv/Conv2DŠ
+tower0/network/layer11/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer11/conv/BatchNorm/ConstÍ
4tower0/network/layer11/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer11_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer11/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer11_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer11_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer11/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer11/conv/Conv2D:output:04tower0/network/layer11/conv/BatchNorm/Const:output:0<tower0/network/layer11/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer11/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer11/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer11/conv/ReluRelu:tower0/network/layer11/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer11/conv/Reluĺ
7tower0/network/layer12/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer12_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer12/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer12/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer11/conv/Relu:activations:0?tower0/network/layer12/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer12/sepconv/depthwiseŻ
.tower0/network/layer12/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer12/sepconv/BatchNorm/ConstÖ
7tower0/network/layer12/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer12_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer12/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer12_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer12_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer12/sepconv/depthwise:output:07tower0/network/layer12/sepconv/BatchNorm/Const:output:0?tower0/network/layer12/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer12/sepconv/ReluRelu=tower0/network/layer12/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer12/sepconv/ReluÍ
1tower0/network/layer13/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer13_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer13/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer13/conv/Conv2DConv2D1tower0/network/layer12/sepconv/Relu:activations:09tower0/network/layer13/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer13/conv/Conv2DŠ
+tower0/network/layer13/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer13/conv/BatchNorm/ConstÍ
4tower0/network/layer13/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer13_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer13/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer13_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer13_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer13/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer13/conv/Conv2D:output:04tower0/network/layer13/conv/BatchNorm/Const:output:0<tower0/network/layer13/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer13/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer13/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer13/conv/ReluRelu:tower0/network/layer13/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer13/conv/Reluĺ
7tower0/network/layer14/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer14_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer14/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer14/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer13/conv/Relu:activations:0?tower0/network/layer14/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer14/sepconv/depthwiseŻ
.tower0/network/layer14/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer14/sepconv/BatchNorm/ConstÖ
7tower0/network/layer14/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer14_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer14/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer14_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer14_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer14/sepconv/depthwise:output:07tower0/network/layer14/sepconv/BatchNorm/Const:output:0?tower0/network/layer14/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer14/sepconv/ReluRelu=tower0/network/layer14/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer14/sepconv/ReluÍ
1tower0/network/layer15/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer15_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer15/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer15/conv/Conv2DConv2D1tower0/network/layer14/sepconv/Relu:activations:09tower0/network/layer15/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer15/conv/Conv2DŠ
+tower0/network/layer15/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer15/conv/BatchNorm/ConstÍ
4tower0/network/layer15/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer15_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer15/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer15_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer15_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer15/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer15/conv/Conv2D:output:04tower0/network/layer15/conv/BatchNorm/Const:output:0<tower0/network/layer15/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer15/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer15/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer15/conv/ReluRelu:tower0/network/layer15/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer15/conv/Reluĺ
7tower0/network/layer16/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer16_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer16/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer16/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer15/conv/Relu:activations:0?tower0/network/layer16/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer16/sepconv/depthwiseŻ
.tower0/network/layer16/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer16/sepconv/BatchNorm/ConstÖ
7tower0/network/layer16/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer16_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer16/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer16_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer16_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer16/sepconv/depthwise:output:07tower0/network/layer16/sepconv/BatchNorm/Const:output:0?tower0/network/layer16/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer16/sepconv/ReluRelu=tower0/network/layer16/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer16/sepconv/ReluÍ
1tower0/network/layer17/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer17_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer17/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer17/conv/Conv2DConv2D1tower0/network/layer16/sepconv/Relu:activations:09tower0/network/layer17/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer17/conv/Conv2DŠ
+tower0/network/layer17/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer17/conv/BatchNorm/ConstÍ
4tower0/network/layer17/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer17_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer17/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer17_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer17_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer17/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer17/conv/Conv2D:output:04tower0/network/layer17/conv/BatchNorm/Const:output:0<tower0/network/layer17/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer17/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer17/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer17/conv/ReluRelu:tower0/network/layer17/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer17/conv/Reluĺ
7tower0/network/layer18/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer18_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer18/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer18/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer17/conv/Relu:activations:0?tower0/network/layer18/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer18/sepconv/depthwiseŻ
.tower0/network/layer18/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer18/sepconv/BatchNorm/ConstÖ
7tower0/network/layer18/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer18_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer18/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer18_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer18_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer18/sepconv/depthwise:output:07tower0/network/layer18/sepconv/BatchNorm/Const:output:0?tower0/network/layer18/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer18/sepconv/ReluRelu=tower0/network/layer18/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer18/sepconv/ReluÍ
1tower0/network/layer19/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer19_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer19/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer19/conv/Conv2DConv2D1tower0/network/layer18/sepconv/Relu:activations:09tower0/network/layer19/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer19/conv/Conv2DŠ
+tower0/network/layer19/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer19/conv/BatchNorm/ConstÍ
4tower0/network/layer19/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer19_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer19/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer19_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer19_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer19/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer19/conv/Conv2D:output:04tower0/network/layer19/conv/BatchNorm/Const:output:0<tower0/network/layer19/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer19/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer19/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer19/conv/ReluRelu:tower0/network/layer19/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer19/conv/Reluĺ
7tower0/network/layer20/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer20_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer20/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer20/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer19/conv/Relu:activations:0?tower0/network/layer20/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer20/sepconv/depthwiseŻ
.tower0/network/layer20/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer20/sepconv/BatchNorm/ConstÖ
7tower0/network/layer20/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer20_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer20/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer20_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer20_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer20/sepconv/depthwise:output:07tower0/network/layer20/sepconv/BatchNorm/Const:output:0?tower0/network/layer20/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer20/sepconv/ReluRelu=tower0/network/layer20/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer20/sepconv/ReluÍ
1tower0/network/layer21/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer21_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer21/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer21/conv/Conv2DConv2D1tower0/network/layer20/sepconv/Relu:activations:09tower0/network/layer21/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer21/conv/Conv2DŠ
+tower0/network/layer21/conv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?2-
+tower0/network/layer21/conv/BatchNorm/ConstÍ
4tower0/network/layer21/conv/BatchNorm/ReadVariableOpReadVariableOp#network_layer21_conv_batchnorm_beta*
_output_shapes	
:*
dtype026
4tower0/network/layer21/conv/BatchNorm/ReadVariableOpö
Etower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp*network_layer21_conv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02G
Etower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOpţ
Gtower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.network_layer21_conv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02I
Gtower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ź
6tower0/network/layer21/conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3+tower0/network/layer21/conv/Conv2D:output:04tower0/network/layer21/conv/BatchNorm/Const:output:0<tower0/network/layer21/conv/BatchNorm/ReadVariableOp:value:0Mtower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Otower0/network/layer21/conv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 28
6tower0/network/layer21/conv/BatchNorm/FusedBatchNormV3Ă
 tower0/network/layer21/conv/ReluRelu:tower0/network/layer21/conv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer21/conv/Reluĺ
7tower0/network/layer22/sepconv/depthwise/ReadVariableOpReadVariableOp)network_layer22_sepconv_depthwise_weights*'
_output_shapes
:*
dtype029
7tower0/network/layer22/sepconv/depthwise/ReadVariableOpÁ
(tower0/network/layer22/sepconv/depthwiseDepthwiseConv2dNative.tower0/network/layer21/conv/Relu:activations:0?tower0/network/layer22/sepconv/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2*
(tower0/network/layer22/sepconv/depthwiseŻ
.tower0/network/layer22/sepconv/BatchNorm/ConstConst*
_output_shapes	
:*
dtype0*
valueB*  ?20
.tower0/network/layer22/sepconv/BatchNorm/ConstÖ
7tower0/network/layer22/sepconv/BatchNorm/ReadVariableOpReadVariableOp&network_layer22_sepconv_batchnorm_beta*
_output_shapes	
:*
dtype029
7tower0/network/layer22/sepconv/BatchNorm/ReadVariableOp˙
Htower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp-network_layer22_sepconv_batchnorm_moving_mean*
_output_shapes	
:*
dtype02J
Htower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp
Jtower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp1network_layer22_sepconv_batchnorm_moving_variance*
_output_shapes	
:*
dtype02L
Jtower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1Ô
9tower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3FusedBatchNormV31tower0/network/layer22/sepconv/depthwise:output:07tower0/network/layer22/sepconv/BatchNorm/Const:output:0?tower0/network/layer22/sepconv/BatchNorm/ReadVariableOp:value:0Ptower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0Rtower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
is_training( 2;
9tower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3Ě
#tower0/network/layer22/sepconv/ReluRelu=tower0/network/layer22/sepconv/BatchNorm/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer22/sepconv/ReluÍ
1tower0/network/layer23/conv/Conv2D/ReadVariableOpReadVariableOpnetwork_layer23_conv_weights*(
_output_shapes
:*
dtype023
1tower0/network/layer23/conv/Conv2D/ReadVariableOpŁ
"tower0/network/layer23/conv/Conv2DConv2D1tower0/network/layer22/sepconv/Relu:activations:09tower0/network/layer23/conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2$
"tower0/network/layer23/conv/Conv2DÁ
2tower0/network/layer23/conv/BiasAdd/ReadVariableOpReadVariableOpnetwork_layer23_conv_biases*
_output_shapes	
:*
dtype024
2tower0/network/layer23/conv/BiasAdd/ReadVariableOpů
#tower0/network/layer23/conv/BiasAddBiasAdd+tower0/network/layer23/conv/Conv2D:output:0:tower0/network/layer23/conv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tower0/network/layer23/conv/BiasAdd­
,tower0/network/layer24/flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 0  2.
,tower0/network/layer24/flatten/flatten/Const
.tower0/network/layer24/flatten/flatten/ReshapeReshape,tower0/network/layer23/conv/BiasAdd:output:05tower0/network/layer24/flatten/flatten/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`20
.tower0/network/layer24/flatten/flatten/Reshapeż
/tower0/network/layer25/fc/MatMul/ReadVariableOpReadVariableOpnetwork_layer25_fc_weights* 
_output_shapes
:
`*
dtype021
/tower0/network/layer25/fc/MatMul/ReadVariableOpó
 tower0/network/layer25/fc/MatMulMatMul7tower0/network/layer24/flatten/flatten/Reshape:output:07tower0/network/layer25/fc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer25/fc/MatMulť
0tower0/network/layer25/fc/BiasAdd/ReadVariableOpReadVariableOpnetwork_layer25_fc_biases*
_output_shapes	
:*
dtype022
0tower0/network/layer25/fc/BiasAdd/ReadVariableOpę
!tower0/network/layer25/fc/BiasAddBiasAdd*tower0/network/layer25/fc/MatMul:product:08tower0/network/layer25/fc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!tower0/network/layer25/fc/BiasAddŻ
 tower0/network/layer26/embeddingIdentity*tower0/network/layer25/fc/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 tower0/network/layer26/embedding"M
 tower0_network_layer26_embedding)tower0/network/layer26/embedding:output:0*¤
_input_shapes
:˙˙˙˙˙˙˙˙˙`@:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::1 -
+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@


map_while_body_5193$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorÍ
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeç
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItemč
map/while/PartitionedCallPartitionedCall4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__sample_to_features_27862
map/while/PartitionedCallö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder"map/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*
_input_shapes
: : : : : : : 
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
: :

_output_shapes
: :

_output_shapes
: 
n
Ö
__inference___call___6167
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/AssertM
	map/ShapeShapesamples*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2ú
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2!
map/TensorArrayV2/element_shapeŔ
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2É
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeţ
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsamplesBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2#
!map/TensorArrayV2_1/element_shapeĆ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counterć
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
bodyR
map_while_body_5908*
condR
map_while_cond_5907*
output_shapes
: : : : : : 2
	map/whileÁ
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙`   @   26
4map/TensorArrayV2Stack/TensorListStack/element_shape
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStackm
ShapeShape/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
Shape
CastCast/map/TensorArrayV2Stack/TensorListStack:tensor:0*

DstT0*

SrcT0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@2
Castt
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ě
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
mulMulstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ě
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ě
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3
Reshape/shapePackmul:z:0strided_slice_2:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
Reshapeł
StatefulPartitionedCallStatefulPartitionedCallReshape:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2ě
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4q
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Reshape_1/shape/1i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2Ş
Reshape_1/shapePackstrided_slice_4:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshape StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	Reshape_1­
IdentityIdentityReshape_1:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ť
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate
n
Ö
__inference___call___5452
samples
sample_rate
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity˘StatefulPartitionedCall˘assert_equal_1/Assert/Assertg
assert_equal_1/yConst*
_output_shapes
: *
dtype0*
value
B :}2
assert_equal_1/y~
assert_equal_1/EqualEqualsample_rateassert_equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_equal_1/Equall
assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/Rankz
assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_equal_1/range/startz
assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
assert_equal_1/range/deltaˇ
assert_equal_1/rangeRange#assert_equal_1/range/start:output:0assert_equal_1/Rank:output:0#assert_equal_1/range/delta:output:0*
_output_shapes
: 2
assert_equal_1/range
assert_equal_1/AllAllassert_equal_1/Equal:z:0assert_equal_1/range:output:0*
_output_shapes
: 2
assert_equal_1/AllŃ
assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2
assert_equal_1/Assert/ConstŞ
assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2
assert_equal_1/Assert/Const_1
assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2
assert_equal_1/Assert/Const_2
assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2
assert_equal_1/Assert/Const_3á
#assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*g
value^B\ BVSample rate must be 16kHz. Instead, was Tensor("sample_rate:0", shape=(), dtype=int32)2%
#assert_equal_1/Assert/Assert/data_0ś
#assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2%
#assert_equal_1/Assert/Assert/data_1
#assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*%
valueB Bx (sample_rate:0) = 2%
#assert_equal_1/Assert/Assert/data_2¤
#assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0**
value!B By (assert_equal_1/y:0) = 2%
#assert_equal_1/Assert/Assert/data_4ę
assert_equal_1/Assert/AssertAssertassert_equal_1/All:output:0,assert_equal_1/Assert/Assert/data_0:output:0,assert_equal_1/Assert/Assert/data_1:output:0,assert_equal_1/Assert/Assert/data_2:output:0sample_rate,assert_equal_1/Assert/Assert/data_4:output:0assert_equal_1/y:output:0*
T

2*
_output_shapes
 2
assert_equal_1/Assert/AssertM
	map/ShapeShapesamples*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2ú
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2!
map/TensorArrayV2/element_shapeŔ
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2É
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeţ
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsamplesBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2#
!map/TensorArrayV2_1/element_shapeĆ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counterć
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
bodyR
map_while_body_5193*
condR
map_while_cond_5192*
output_shapes
: : : : : : 2
	map/whileÁ
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙`   @   26
4map/TensorArrayV2Stack/TensorListStack/element_shape
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStackm
ShapeShape/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:2
Shape
CastCast/map/TensorArrayV2Stack/TensorListStack:tensor:0*

DstT0*

SrcT0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`@2
Castt
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ě
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
mulMulstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ě
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ě
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3
Reshape/shapePackmul:z:0strided_slice_2:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
Reshapeł
StatefulPartitionedCallStatefulPartitionedCallReshape:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_read_only_resource_inputsb
`^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_23302
StatefulPartitionedCallx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2ě
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4q
Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Reshape_1/shape/1i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2Ş
Reshape_1/shapePackstrided_slice_4:output:0Reshape_1/shape/1:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshape StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	Reshape_1­
IdentityIdentityReshape_1:output:0^StatefulPartitionedCall^assert_equal_1/Assert/Assert*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ť
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2<
assert_equal_1/Assert/Assertassert_equal_1/Assert/Assert:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	samples:C?

_output_shapes
: 
%
_user_specified_namesample_rate
ç'
Đ!
__inference_pruned_1931
network_layer1_conv_weights&
"network_layer1_conv_batchnorm_beta-
)network_layer1_conv_batchnorm_moving_mean1
-network_layer1_conv_batchnorm_moving_variance,
(network_layer2_sepconv_depthwise_weights)
%network_layer2_sepconv_batchnorm_beta0
,network_layer2_sepconv_batchnorm_moving_mean4
0network_layer2_sepconv_batchnorm_moving_variance
network_layer3_conv_weights&
"network_layer3_conv_batchnorm_beta-
)network_layer3_conv_batchnorm_moving_mean1
-network_layer3_conv_batchnorm_moving_variance,
(network_layer4_sepconv_depthwise_weights)
%network_layer4_sepconv_batchnorm_beta0
,network_layer4_sepconv_batchnorm_moving_mean4
0network_layer4_sepconv_batchnorm_moving_variance
network_layer5_conv_weights&
"network_layer5_conv_batchnorm_beta-
)network_layer5_conv_batchnorm_moving_mean1
-network_layer5_conv_batchnorm_moving_variance,
(network_layer6_sepconv_depthwise_weights)
%network_layer6_sepconv_batchnorm_beta0
,network_layer6_sepconv_batchnorm_moving_mean4
0network_layer6_sepconv_batchnorm_moving_variance
network_layer7_conv_weights&
"network_layer7_conv_batchnorm_beta-
)network_layer7_conv_batchnorm_moving_mean1
-network_layer7_conv_batchnorm_moving_variance,
(network_layer8_sepconv_depthwise_weights)
%network_layer8_sepconv_batchnorm_beta0
,network_layer8_sepconv_batchnorm_moving_mean4
0network_layer8_sepconv_batchnorm_moving_variance
network_layer9_conv_weights&
"network_layer9_conv_batchnorm_beta-
)network_layer9_conv_batchnorm_moving_mean1
-network_layer9_conv_batchnorm_moving_variance-
)network_layer10_sepconv_depthwise_weights*
&network_layer10_sepconv_batchnorm_beta1
-network_layer10_sepconv_batchnorm_moving_mean5
1network_layer10_sepconv_batchnorm_moving_variance 
network_layer11_conv_weights'
#network_layer11_conv_batchnorm_beta.
*network_layer11_conv_batchnorm_moving_mean2
.network_layer11_conv_batchnorm_moving_variance-
)network_layer12_sepconv_depthwise_weights*
&network_layer12_sepconv_batchnorm_beta1
-network_layer12_sepconv_batchnorm_moving_mean5
1network_layer12_sepconv_batchnorm_moving_variance 
network_layer13_conv_weights'
#network_layer13_conv_batchnorm_beta.
*network_layer13_conv_batchnorm_moving_mean2
.network_layer13_conv_batchnorm_moving_variance-
)network_layer14_sepconv_depthwise_weights*
&network_layer14_sepconv_batchnorm_beta1
-network_layer14_sepconv_batchnorm_moving_mean5
1network_layer14_sepconv_batchnorm_moving_variance 
network_layer15_conv_weights'
#network_layer15_conv_batchnorm_beta.
*network_layer15_conv_batchnorm_moving_mean2
.network_layer15_conv_batchnorm_moving_variance-
)network_layer16_sepconv_depthwise_weights*
&network_layer16_sepconv_batchnorm_beta1
-network_layer16_sepconv_batchnorm_moving_mean5
1network_layer16_sepconv_batchnorm_moving_variance 
network_layer17_conv_weights'
#network_layer17_conv_batchnorm_beta.
*network_layer17_conv_batchnorm_moving_mean2
.network_layer17_conv_batchnorm_moving_variance-
)network_layer18_sepconv_depthwise_weights*
&network_layer18_sepconv_batchnorm_beta1
-network_layer18_sepconv_batchnorm_moving_mean5
1network_layer18_sepconv_batchnorm_moving_variance 
network_layer19_conv_weights'
#network_layer19_conv_batchnorm_beta.
*network_layer19_conv_batchnorm_moving_mean2
.network_layer19_conv_batchnorm_moving_variance-
)network_layer20_sepconv_depthwise_weights*
&network_layer20_sepconv_batchnorm_beta1
-network_layer20_sepconv_batchnorm_moving_mean5
1network_layer20_sepconv_batchnorm_moving_variance 
network_layer21_conv_weights'
#network_layer21_conv_batchnorm_beta.
*network_layer21_conv_batchnorm_moving_mean2
.network_layer21_conv_batchnorm_moving_variance-
)network_layer22_sepconv_depthwise_weights*
&network_layer22_sepconv_batchnorm_beta1
-network_layer22_sepconv_batchnorm_moving_mean5
1network_layer22_sepconv_batchnorm_moving_variance 
network_layer23_conv_weights
network_layer23_conv_biases
network_layer25_fc_weights
network_layer25_fc_biases
network_layer28_fc_weights
network_layer28_fc_biases
dummy_fetch˘
group_deps*
initNoOp*
_output_shapes
 2
init.
init_1NoOp*
_output_shapes
 2
init_1@
init_all_tablesNoOp*
_output_shapes
 2
init_all_tablesX

group_depsNoOp^init^init_1^init_all_tables*
_output_shapes
 2

group_depsI
dummy_fetch_0Const*
dtype0*
valueB
 *    2
dummy_fetch"%
dummy_fetchdummy_fetch_0:output:0*
_input_shapesű
ř::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2

group_deps
group_deps
ŮĆ
;
$__inference__sample_to_features_4670
x
identityT
CastCastx*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Cast[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ţ˙F2
	truediv/yi
truedivRealDivCast:y:0truediv/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivB
SizeSizetruediv:z:0*
T0*
_output_shapes
: 2
SizeS
Less/yConst*
_output_shapes
: *
dtype0*
value
B :}2
Less/yU
LessLessSize:output:0Less/y:output:0*
T0*
_output_shapes
: 2
Lessż
condStatelessIfLess:z:0Size:output:0truediv:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *"
else_branchR
cond_false_4242*"
output_shapes
:˙˙˙˙˙˙˙˙˙*!
then_branchR
cond_true_42412
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identitym
Cast_1Castcond/Identity:output:0*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Cast_1i
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/frame_lengthe
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B : 2
stft/frame_stepe
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2
stft/fft_lengthm
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
stft/frame/axis^
stft/frame/ShapeShape
Cast_1:y:0*
T0*
_output_shapes
:2
stft/frame/Shaped
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Rankr
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range/startr
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range/deltaĽ
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:2
stft/frame/range
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
stft/frame/strided_slice/stack
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 stft/frame/strided_slice/stack_1
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 stft/frame/strided_slice/stack_2¤
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
stft/frame/strided_slicef
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/sub/y}
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft/frame/sub
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_1l
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/packed/1ł
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/packedz
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/split/split_dimÔ
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*
T0*

Tlen0*"
_output_shapes
: :: *
	num_split2
stft/frame/splitw
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape{
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
stft/frame/Reshape/shape_1
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
stft/frame/Reshaped
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Sizeh
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Size_1
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: 2
stft/frame/sub_2
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: 2
stft/frame/floordivf
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/add/x~
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft/frame/addn
stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/Maximum/x
stft/frame/MaximumMaximumstft/frame/Maximum/x:output:0stft/frame/add:z:0*
T0*
_output_shapes
: 2
stft/frame/Maximumn
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/gcd/Constt
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_1/y
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_1t
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_2/y
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_2t
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/floordiv_3/y
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
stft/frame/floordiv_3f
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :P2
stft/frame/mul/y~
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2
stft/frame/mul
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:2
stft/frame/concat/values_1r
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat/axisÜ
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P2 
stft/frame/concat_1/values_1/1ś
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1/values_1v
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_1/axisä
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_1x
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
stft/frame/zeros_like
stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
stft/frame/ones_like/Shapez
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/ones_like/ConstŁ
stft/frame/ones_likeFill#stft/frame/ones_like/Shape:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
stft/frame/ones_likeä
stft/frame/StridedSliceStridedSlice
Cast_1:y:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/StridedSliceŠ
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/Reshape_1v
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_1/startv
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_1/delta´
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/range_1
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/mul_1~
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_2/shape/1­
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_2/shape¤
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_2v
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/range_2/startv
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/range_2/deltaŽ
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
:2
stft/frame/range_2~
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
stft/frame/Reshape_3/shape/0°
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
stft/frame/Reshape_3/shape˘
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2
stft/frame/Reshape_3
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/add_1ç
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙P2
stft/frame/GatherV2Ś
stft/frame/concat_2/values_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2/values_1v
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/frame/concat_2/axisä
stft/frame/concat_2ConcatV2stft/frame/split:output:0%stft/frame/concat_2/values_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
stft/frame/concat_2Ś
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/frame/Reshape_4x
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
stft/hann_window/periodic
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2
stft/hann_window/Cast|
stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/FloorMod/yĽ
stft/hann_window/FloorModFloorModstft/frame_length:output:0$stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/FloorModr
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub/x
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2
stft/hann_window/sub
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2
stft/hann_window/mul
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2
stft/hann_window/addv
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/sub_1/y
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2
stft/hann_window/sub_1
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
stft/hann_window/Cast_1~
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
stft/hann_window/range/start~
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
stft/hann_window/range/deltaŔ
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:2
stft/hann_window/range
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2
stft/hann_window/Cast_2y
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-DTű!@2
stft/hann_window/Const
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_1
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2
stft/hann_window/truedivw
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2
stft/hann_window/Cos}
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/mul_2/x
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2
stft/hann_window/mul_2}
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      ŕ?2
stft/hann_window/sub_2/x
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2
stft/hann_window/sub_2
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

stft/mult
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:2
stft/rfft/packed
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   2
stft/rfft/Pad/paddings
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stft/rfft/Padw
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2
stft/rfft/fft_length
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*
Tcomplex0*
Treal0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	stft/rfftk
Abs
ComplexAbsstft/rfft:output:0*
T0*

Tout0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Abs
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :}2+
)linear_to_mel_weight_matrix/sample_rate/xž
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'linear_to_mel_weight_matrix/sample_rateĽ
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     @_@2.
,linear_to_mel_weight_matrix/lower_edge_hertzĽ
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB 2     L˝@2.
,linear_to_mel_weight_matrix/upper_edge_hertz
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2#
!linear_to_mel_weight_matrix/Const
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @2'
%linear_to_mel_weight_matrix/truediv/yÓ
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2%
#linear_to_mel_weight_matrix/truediv
(linear_to_mel_weight_matrix/linspace/numConst*
_output_shapes
: *
dtype0*
value
B :2*
(linear_to_mel_weight_matrix/linspace/numÁ
)linear_to_mel_weight_matrix/linspace/CastCast1linear_to_mel_weight_matrix/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/linspace/CastÁ
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_1
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2,
*linear_to_mel_weight_matrix/linspace/Shape
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace/Shape_1˙
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastArgsů
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace/BroadcastToú
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1Ź
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/ExpandDims/dim
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/ExpandDims°
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace/ExpandDims_1Ś
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_2Ś
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2.
,linear_to_mel_weight_matrix/linspace/Shape_3ž
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8linear_to_mel_weight_matrix/linspace/strided_slice/stackÂ
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Â
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Â
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2linear_to_mel_weight_matrix/linspace/strided_slice
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2,
*linear_to_mel_weight_matrix/linspace/add/yđ
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/add´
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z29
7linear_to_mel_weight_matrix/linspace/SelectV2/condition¤
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/linspace/SelectV2/tľ
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace/SelectV2
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/linspace/sub/yŕ
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: 2*
(linear_to_mel_weight_matrix/linspace/sub˘
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 20
.linear_to_mel_weight_matrix/linspace/Maximum/yď
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace/Maximum
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace/sub_1/yć
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace/sub_1Ś
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :22
0linear_to_mel_weight_matrix/linspace/Maximum_1/y÷
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace/Maximum_1ú
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/sub_2Ć
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace/Cast_2í
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/truedivŹ
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3linear_to_mel_weight_matrix/linspace/GreaterEqual/y
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace/GreaterEqualą
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eś
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/SelectV2_1Ś
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/startŚ
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0linear_to_mel_weight_matrix/linspace/range/deltaÔ
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace/range/Castľ
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/rangeĚ
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:˙2-
+linear_to_mel_weight_matrix/linspace/Cast_3Ş
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2linear_to_mel_weight_matrix/linspace/range_1/startŞ
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace/range_1/delta¸
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace/range_1ő
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/linspace/Equal¨
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eą
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_2ř
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:˙2.
,linear_to_mel_weight_matrix/linspace/Reshapeę
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:˙2*
(linear_to_mel_weight_matrix/linspace/mulď
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:˙2,
*linear_to_mel_weight_matrix/linspace/add_1ó
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:2-
+linear_to_mel_weight_matrix/linspace/concatŹ
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 21
/linear_to_mel_weight_matrix/linspace/zeros_likeŠ
/linear_to_mel_weight_matrix/linspace/SelectV2_3SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:0-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/Shape_2:output:0*
T0*
_output_shapes
:21
/linear_to_mel_weight_matrix/linspace/SelectV2_3ž
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:08linear_to_mel_weight_matrix/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes	
:2,
*linear_to_mel_weight_matrix/linspace/SliceŹ
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/linear_to_mel_weight_matrix/strided_slice/stack°
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/strided_slice/stack_1°
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/strided_slice/stack_2
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask2+
)linear_to_mel_weight_matrix/strided_sliceą
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@24
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/y
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:22
0linear_to_mel_weight_matrix/hertz_to_mel/truedivŠ
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?20
.linear_to_mel_weight_matrix/hertz_to_mel/add/xú
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/addť
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/LogŠ
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @20
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xô
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:2.
,linear_to_mel_weight_matrix/hertz_to_mel/mul
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*linear_to_mel_weight_matrix/ExpandDims/dimď
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	2(
&linear_to_mel_weight_matrix/ExpandDimsľ
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_1/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/addź
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulľ
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2     ŕ@26
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/hertz_to_mel_2/truediv­
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB 2      đ?22
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xý
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/addź
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/Log­
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2     @22
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x÷
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/hertz_to_mel_2/mul
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :B2,
*linear_to_mel_weight_matrix/linspace_1/numÇ
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+linear_to_mel_weight_matrix/linspace_1/CastÇ
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_1
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/linspace_1/ShapeŁ
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/linspace_1/Shape_1
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgs
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 24
2linear_to_mel_weight_matrix/linspace_1/BroadcastTo
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 26
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1°
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/ExpandDims´
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_2Ş
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:20
.linear_to_mel_weight_matrix/linspace_1/Shape_3Â
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackĆ
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Ć
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Î
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4linear_to_mel_weight_matrix/linspace_1/strided_slice
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2.
,linear_to_mel_weight_matrix/linspace_1/add/yř
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/add¸
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z2;
9linear_to_mel_weight_matrix/linspace_1/SelectV2/condition¨
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tż
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: 21
/linear_to_mel_weight_matrix/linspace_1/SelectV2
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/linspace_1/sub/yč
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/linspace_1/subŚ
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 22
0linear_to_mel_weight_matrix/linspace_1/Maximum/y÷
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: 20
.linear_to_mel_weight_matrix/linspace_1/Maximum˘
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/linspace_1/sub_1/yî
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/linspace_1/sub_1Ş
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :24
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/y˙
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: 22
0linear_to_mel_weight_matrix/linspace_1/Maximum_1
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/sub_2Ě
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-linear_to_mel_weight_matrix/linspace_1/Cast_2ő
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/truediv°
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 27
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: 25
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualľ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eŔ
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1Ş
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/startŞ
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2linear_to_mel_weight_matrix/linspace_1/range/deltaÚ
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 23
1linear_to_mel_weight_matrix/linspace_1/range/Castž
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/rangeŃ
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:@2/
-linear_to_mel_weight_matrix/linspace_1/Cast_3Ž
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4linear_to_mel_weight_matrix/linspace_1/range_1/startŽ
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaÂ
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:20
.linear_to_mel_weight_matrix/linspace_1/range_1ý
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:2.
,linear_to_mel_weight_matrix/linspace_1/EqualŹ
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eť
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2˙
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:@20
.linear_to_mel_weight_matrix/linspace_1/Reshapeń
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:@2,
*linear_to_mel_weight_matrix/linspace_1/mulö
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:@2.
,linear_to_mel_weight_matrix/linspace_1/add_1ţ
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:B2/
-linear_to_mel_weight_matrix/linspace_1/concat°
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 23
1linear_to_mel_weight_matrix/linspace_1/zeros_likeł
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:0/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/linspace_1/SelectV2_3Ç
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:B2.
,linear_to_mel_weight_matrix/linspace_1/Slice˘
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/frame_length
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2.
,linear_to_mel_weight_matrix/frame/frame_step
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2(
&linear_to_mel_weight_matrix/frame/axis
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:B2)
'linear_to_mel_weight_matrix/frame/Shape
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB 2.
,linear_to_mel_weight_matrix/frame/Size/Const
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2(
&linear_to_mel_weight_matrix/frame/SizeŁ
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 20
.linear_to_mel_weight_matrix/frame/Size_1/Const
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(linear_to_mel_weight_matrix/frame/Size_1
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :B2)
'linear_to_mel_weight_matrix/frame/sub/xá
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/subç
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: 2,
*linear_to_mel_weight_matrix/frame/floordiv
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'linear_to_mel_weight_matrix/frame/add/xÚ
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: 2'
%linear_to_mel_weight_matrix/frame/add
+linear_to_mel_weight_matrix/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2-
+linear_to_mel_weight_matrix/frame/Maximum/xă
)linear_to_mel_weight_matrix/frame/MaximumMaximum4linear_to_mel_weight_matrix/frame/Maximum/x:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: 2+
)linear_to_mel_weight_matrix/frame/Maximum
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/frame/gcd/Const˘
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_1/yű
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_1˘
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :20
.linear_to_mel_weight_matrix/frame/floordiv_2/yů
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2.
,linear_to_mel_weight_matrix/frame/floordiv_2Š
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_0°
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:B23
1linear_to_mel_weight_matrix/frame/concat/values_1Š
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 23
1linear_to_mel_weight_matrix/frame/concat/values_2 
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/concat/axisú
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(linear_to_mel_weight_matrix/frame/concat­
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_0ť
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"B      25
3linear_to_mel_weight_matrix/frame/concat_1/values_1­
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_1/values_2¤
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_1/axis
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_1´
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:B25
3linear_to_mel_weight_matrix/frame/zeros_like/tensorŚ
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2.
,linear_to_mel_weight_matrix/frame/zeros_like°
1linear_to_mel_weight_matrix/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:23
1linear_to_mel_weight_matrix/frame/ones_like/Shape¨
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :23
1linear_to_mel_weight_matrix/frame/ones_like/Const˙
+linear_to_mel_weight_matrix/frame/ones_likeFill:linear_to_mel_weight_matrix/frame/ones_like/Shape:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2-
+linear_to_mel_weight_matrix/frame/ones_likeů
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:B20
.linear_to_mel_weight_matrix/frame/StridedSliceř
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:B2+
)linear_to_mel_weight_matrix/frame/Reshape 
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-linear_to_mel_weight_matrix/frame/range/start 
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-linear_to_mel_weight_matrix/frame/range/delta
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:@2)
'linear_to_mel_weight_matrix/frame/rangeŢ
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:@2'
%linear_to_mel_weight_matrix/frame/mulŹ
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_1/shapeő
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_1¤
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/range_1/start¤
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/linear_to_mel_weight_matrix/frame/range_1/deltaĄ
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:2+
)linear_to_mel_weight_matrix/frame/range_1Ź
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :25
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:23
1linear_to_mel_weight_matrix/frame/Reshape_2/shapeţ
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:2-
+linear_to_mel_weight_matrix/frame/Reshape_2đ
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:@2)
'linear_to_mel_weight_matrix/frame/add_1¤
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/GatherV2/axisĎ
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:@2,
*linear_to_mel_weight_matrix/frame/GatherV2­
3linear_to_mel_weight_matrix/frame/concat_2/values_0Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_0
3linear_to_mel_weight_matrix/frame/concat_2/values_1Pack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:25
3linear_to_mel_weight_matrix/frame/concat_2/values_1­
3linear_to_mel_weight_matrix/frame/concat_2/values_2Const*
_output_shapes
: *
dtype0*
valueB 25
3linear_to_mel_weight_matrix/frame/concat_2/values_2¤
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/linear_to_mel_weight_matrix/frame/concat_2/axis
*linear_to_mel_weight_matrix/frame/concat_2ConcatV2<linear_to_mel_weight_matrix/frame/concat_2/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_2/values_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2,
*linear_to_mel_weight_matrix/frame/concat_2ř
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:@2-
+linear_to_mel_weight_matrix/frame/Reshape_3
#linear_to_mel_weight_matrix/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#linear_to_mel_weight_matrix/Const_1
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+linear_to_mel_weight_matrix/split/split_dim
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:@:@:@*
	num_split2#
!linear_to_mel_weight_matrix/split§
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2+
)linear_to_mel_weight_matrix/Reshape/shapeŢ
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:@2%
#linear_to_mel_weight_matrix/ReshapeŤ
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_1/shapeä
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_1Ť
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2-
+linear_to_mel_weight_matrix/Reshape_2/shapeä
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:@2'
%linear_to_mel_weight_matrix/Reshape_2Ň
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	@2!
linear_to_mel_weight_matrix/subÔ
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_1Ď
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_1Ř
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	@2#
!linear_to_mel_weight_matrix/sub_2Ö
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:@2#
!linear_to_mel_weight_matrix/sub_3Ń
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	@2'
%linear_to_mel_weight_matrix/truediv_2Ő
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/MinimumÔ
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	@2%
#linear_to_mel_weight_matrix/Maximum­
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2&
$linear_to_mel_weight_matrix/paddingsĂ
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	@2
linear_to_mel_weight_matrix{
matmulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
matmul_
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB 2ę-q=2
	Maximum/yu
MaximumMaximummatmul:product:0Maximum/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
MaximumW
add/yConst*
_output_shapes
: *
dtype0*
valueB 2üŠńŇMbP?2
add/yb
addAddV2Maximum:z:0add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
addL
LogLogadd:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Logj
frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :`2
frame/frame_lengthf
frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2
frame/frame_stepZ

frame/axisConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/axisQ
frame/ShapeShapeLog:y:0*
T0*
_output_shapes
:2
frame/ShapeZ

frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2

frame/Rankh
frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range/starth
frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range/delta
frame/rangeRangeframe/range/start:output:0frame/Rank:output:0frame/range/delta:output:0*
_output_shapes
:2
frame/range
frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/strided_slice/stack
frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_1
frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
frame/strided_slice/stack_2
frame/strided_sliceStridedSliceframe/range:output:0"frame/strided_slice/stack:output:0$frame/strided_slice/stack_1:output:0$frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
frame/strided_slice\
frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/sub/yi
	frame/subSubframe/Rank:output:0frame/sub/y:output:0*
T0*
_output_shapes
: 2
	frame/subo
frame/sub_1Subframe/sub:z:0frame/strided_slice:output:0*
T0*
_output_shapes
: 2
frame/sub_1b
frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/packed/1
frame/packedPackframe/strided_slice:output:0frame/packed/1:output:0frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
frame/packedp
frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/split/split_dim˝
frame/splitSplitVframe/Shape:output:0frame/packed:output:0frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
: ::*
	num_split2
frame/splitm
frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shapeq
frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
frame/Reshape/shape_1
frame/ReshapeReshapeframe/split:output:1frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2
frame/ReshapeZ

frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2

frame/Size^
frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Size_1w
frame/sub_2Subframe/Reshape:output:0frame/frame_length:output:0*
T0*
_output_shapes
: 2
frame/sub_2y
frame/floordivFloorDivframe/sub_2:z:0frame/frame_step:output:0*
T0*
_output_shapes
: 2
frame/floordiv\
frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
frame/add/xj
	frame/addAddV2frame/add/x:output:0frame/floordiv:z:0*
T0*
_output_shapes
: 2
	frame/addd
frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/Maximum/xs
frame/MaximumMaximumframe/Maximum/x:output:0frame/add:z:0*
T0*
_output_shapes
: 2
frame/Maximumd
frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/gcd/Constj
frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_1/y
frame/floordiv_1FloorDivframe/frame_length:output:0frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_1j
frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_2/y
frame/floordiv_2FloorDivframe/frame_step:output:0frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_2j
frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/floordiv_3/y
frame/floordiv_3FloorDivframe/Reshape:output:0frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2
frame/floordiv_3\
frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
frame/mul/yj
	frame/mulMulframe/floordiv_3:z:0frame/mul/y:output:0*
T0*
_output_shapes
: 2
	frame/muls
frame/concat/values_1Packframe/mul:z:0*
N*
T0*
_output_shapes
:2
frame/concat/values_1h
frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat/axisž
frame/concatConcatV2frame/split:output:0frame/concat/values_1:output:0frame/split:output:2frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concatx
frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/concat_1/values_1/1˘
frame/concat_1/values_1Packframe/floordiv_3:z:0"frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1/values_1l
frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_1/axisĆ
frame/concat_1ConcatV2frame/split:output:0 frame/concat_1/values_1:output:0frame/split:output:2frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_1n
frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2
frame/zeros_likex
frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
frame/ones_like/Shapep
frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
frame/ones_like/Const
frame/ones_likeFillframe/ones_like/Shape:output:0frame/ones_like/Const:output:0*
T0*
_output_shapes
:2
frame/ones_likeŐ
frame/StridedSliceStridedSliceLog:y:0frame/zeros_like:output:0frame/concat:output:0frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/StridedSlice˘
frame/Reshape_1Reshapeframe/StridedSlice:output:0frame/concat_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
frame/Reshape_1l
frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_1/startl
frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_1/delta
frame/range_1Rangeframe/range_1/start:output:0frame/Maximum:z:0frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/range_1}
frame/mul_1Mulframe/range_1:output:0frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/mul_1t
frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_2/shape/1
frame/Reshape_2/shapePackframe/Maximum:z:0 frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2
frame/Reshape_2/shape
frame/Reshape_2Reshapeframe/mul_1:z:0frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
frame/Reshape_2l
frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/range_2/startl
frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
frame/range_2/delta
frame/range_2Rangeframe/range_2/start:output:0frame/floordiv_1:z:0frame/range_2/delta:output:0*
_output_shapes
:`2
frame/range_2t
frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
frame/Reshape_3/shape/0
frame/Reshape_3/shapePack frame/Reshape_3/shape/0:output:0frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2
frame/Reshape_3/shape
frame/Reshape_3Reshapeframe/range_2:output:0frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:`2
frame/Reshape_3
frame/add_1AddV2frame/Reshape_2:output:0frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`2
frame/add_1Ű
frame/GatherV2GatherV2frame/Reshape_1:output:0frame/add_1:z:0frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙`˙˙˙˙˙˙˙˙˙2
frame/GatherV2
frame/concat_2/values_1Packframe/Maximum:z:0frame/frame_length:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2/values_1l
frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
frame/concat_2/axisĆ
frame/concat_2ConcatV2frame/split:output:0 frame/concat_2/values_1:output:0frame/split:output:2frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2
frame/concat_2
frame/Reshape_4Reshapeframe/GatherV2:output:0frame/concat_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
frame/Reshape_4p
IdentityIdentityframe/Reshape_4:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2

Identity"
identityIdentity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Đ

__inference__initializer_6869
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

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identityö
PartitionedCallPartitionedCallunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*i
Tinb
`2^*
Tout
2*
_output_shapes
: *
_read_only_resource_inputsb
`^ 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_19312
PartitionedCallP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapesű
ř::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 

map_while_cond_5192$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_5192___redundant_placeholder0
map_while_identity

map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*!
_input_shapes
: : : : : :: 
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
: :

_output_shapes
: :

_output_shapes
:"ÔJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ďK

trill_module
	variables
trainable_variables
save_counter

signatures
f__call__"
_generic_user_object
_
initializer
asset_paths

signatures
	variables"
_generic_user_object

0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93"
trackable_list_wrapper
:	 2save_counter
"
signature_map
@
g_create_resource
h_initialize
i_destroy_resourceR 
 "
trackable_list_wrapper
&
j	inference"
signature_map
5:3 2network/layer1/conv/weights
0:. 2"network/layer1/conv/BatchNorm/beta
5:3 2)network/layer1/conv/BatchNorm/moving_mean
9:7 2-network/layer1/conv/BatchNorm/moving_variance
B:@ 2(network/layer2/sepconv/depthwise_weights
3:1 2%network/layer2/sepconv/BatchNorm/beta
8:6 2,network/layer2/sepconv/BatchNorm/moving_mean
<:: 20network/layer2/sepconv/BatchNorm/moving_variance
5:3 @2network/layer3/conv/weights
0:.@2"network/layer3/conv/BatchNorm/beta
5:3@2)network/layer3/conv/BatchNorm/moving_mean
9:7@2-network/layer3/conv/BatchNorm/moving_variance
B:@@2(network/layer4/sepconv/depthwise_weights
3:1@2%network/layer4/sepconv/BatchNorm/beta
8:6@2,network/layer4/sepconv/BatchNorm/moving_mean
<::@20network/layer4/sepconv/BatchNorm/moving_variance
6:4@2network/layer5/conv/weights
1:/2"network/layer5/conv/BatchNorm/beta
6:42)network/layer5/conv/BatchNorm/moving_mean
::82-network/layer5/conv/BatchNorm/moving_variance
C:A2(network/layer6/sepconv/depthwise_weights
4:22%network/layer6/sepconv/BatchNorm/beta
9:72,network/layer6/sepconv/BatchNorm/moving_mean
=:;20network/layer6/sepconv/BatchNorm/moving_variance
7:52network/layer7/conv/weights
1:/2"network/layer7/conv/BatchNorm/beta
6:42)network/layer7/conv/BatchNorm/moving_mean
::82-network/layer7/conv/BatchNorm/moving_variance
C:A2(network/layer8/sepconv/depthwise_weights
4:22%network/layer8/sepconv/BatchNorm/beta
9:72,network/layer8/sepconv/BatchNorm/moving_mean
=:;20network/layer8/sepconv/BatchNorm/moving_variance
7:52network/layer9/conv/weights
1:/2"network/layer9/conv/BatchNorm/beta
6:42)network/layer9/conv/BatchNorm/moving_mean
::82-network/layer9/conv/BatchNorm/moving_variance
D:B2)network/layer10/sepconv/depthwise_weights
5:32&network/layer10/sepconv/BatchNorm/beta
::82-network/layer10/sepconv/BatchNorm/moving_mean
>:<21network/layer10/sepconv/BatchNorm/moving_variance
8:62network/layer11/conv/weights
2:02#network/layer11/conv/BatchNorm/beta
7:52*network/layer11/conv/BatchNorm/moving_mean
;:92.network/layer11/conv/BatchNorm/moving_variance
D:B2)network/layer12/sepconv/depthwise_weights
5:32&network/layer12/sepconv/BatchNorm/beta
::82-network/layer12/sepconv/BatchNorm/moving_mean
>:<21network/layer12/sepconv/BatchNorm/moving_variance
8:62network/layer13/conv/weights
2:02#network/layer13/conv/BatchNorm/beta
7:52*network/layer13/conv/BatchNorm/moving_mean
;:92.network/layer13/conv/BatchNorm/moving_variance
D:B2)network/layer14/sepconv/depthwise_weights
5:32&network/layer14/sepconv/BatchNorm/beta
::82-network/layer14/sepconv/BatchNorm/moving_mean
>:<21network/layer14/sepconv/BatchNorm/moving_variance
8:62network/layer15/conv/weights
2:02#network/layer15/conv/BatchNorm/beta
7:52*network/layer15/conv/BatchNorm/moving_mean
;:92.network/layer15/conv/BatchNorm/moving_variance
D:B2)network/layer16/sepconv/depthwise_weights
5:32&network/layer16/sepconv/BatchNorm/beta
::82-network/layer16/sepconv/BatchNorm/moving_mean
>:<21network/layer16/sepconv/BatchNorm/moving_variance
8:62network/layer17/conv/weights
2:02#network/layer17/conv/BatchNorm/beta
7:52*network/layer17/conv/BatchNorm/moving_mean
;:92.network/layer17/conv/BatchNorm/moving_variance
D:B2)network/layer18/sepconv/depthwise_weights
5:32&network/layer18/sepconv/BatchNorm/beta
::82-network/layer18/sepconv/BatchNorm/moving_mean
>:<21network/layer18/sepconv/BatchNorm/moving_variance
8:62network/layer19/conv/weights
2:02#network/layer19/conv/BatchNorm/beta
7:52*network/layer19/conv/BatchNorm/moving_mean
;:92.network/layer19/conv/BatchNorm/moving_variance
D:B2)network/layer20/sepconv/depthwise_weights
5:32&network/layer20/sepconv/BatchNorm/beta
::82-network/layer20/sepconv/BatchNorm/moving_mean
>:<21network/layer20/sepconv/BatchNorm/moving_variance
8:62network/layer21/conv/weights
2:02#network/layer21/conv/BatchNorm/beta
7:52*network/layer21/conv/BatchNorm/moving_mean
;:92.network/layer21/conv/BatchNorm/moving_variance
D:B2)network/layer22/sepconv/depthwise_weights
5:32&network/layer22/sepconv/BatchNorm/beta
::82-network/layer22/sepconv/BatchNorm/moving_mean
>:<21network/layer22/sepconv/BatchNorm/moving_variance
8:62network/layer23/conv/weights
*:(2network/layer23/conv/biases
.:,
`2network/layer25/fc/weights
(:&2network/layer25/fc/biases
.:,
`2network/layer28/fc/weights
(:&`2network/layer28/fc/biases
Ú2×
__inference___call___5452
__inference___call___5876
__inference___call___6458
__inference___call___6670
__inference___call___5664
__inference___call___6167˛
Š˛Ľ
FullArgSpec-
args%"
jself
	jsamples
jsample_rate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
°2­
__inference__creator_6675
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
´2ą
__inference__initializer_6869
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
˛2Ż
__inference__destroyer_6874
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
.B,
__inference_pruned_2330INFERENCE_INPUT
__inference___call___5452ö^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeO˘L
E˘B
*'
samples˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "CŞ@
>
	embedding1.
	embedding˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ú
__inference___call___5664Ü^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeB˘?
8˘5

samples˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "6Ş3
1
	embedding$!
	embedding˙˙˙˙˙˙˙˙˙ú
__inference___call___5876Ü^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeB˘?
8˘5

samples˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "6Ş3
1
	embedding$!
	embedding˙˙˙˙˙˙˙˙˙
__inference___call___6167ö^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeO˘L
E˘B
*'
samples˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "CŞ@
>
	embedding1.
	embedding˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
__inference___call___6458ö^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeO˘L
E˘B
*'
samples˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "CŞ@
>
	embedding1.
	embedding˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ú
__inference___call___6670Ü^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdeB˘?
8˘5

samples˙˙˙˙˙˙˙˙˙

sample_rate 
Ş "6Ş3
1
	embedding$!
	embedding˙˙˙˙˙˙˙˙˙5
__inference__creator_6675˘

˘ 
Ş " 7
__inference__destroyer_6874˘

˘ 
Ş " 
__inference__initializer_6869x^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcde˘

˘ 
Ş " ć
__inference_pruned_2330Ę^	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcde
 "dŞa
_
 tower0/network/layer26/embedding;8
 tower0/network/layer26/embedding˙˙˙˙˙˙˙˙˙