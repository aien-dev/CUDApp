ClearAll[hh, mat, vec, uu, mm];
hh=Nest[KroneckerProduct[#,{{1,1},{1,-1}}]&,{{1}},7];
small=Nest[KroneckerProduct[#,{{1,1},{1,-1}}]&,{{1}},2];
mat=Array[ww,{3,4}]; vec=Array[xx,4];
result=<|"HadamardSquared128I"->(hh.hh==128 IdentityMatrix[128]),
"FoldedLinearIdentity"->(Expand[((mat.small).(small.vec))/4-mat.vec]=={0,0,0}),
"NormalizedQuantizerRatio"->FullSimplify[(7 uu/Sqrt[128])/(mm/Sqrt[128])==7 uu/mm,mm>0],
"MaxFP8Units"->14*2^14,"MaxHadamardUnits"->128*14*2^14,
"MaxNumerator"->128*7*(128*14*2^14),"Denominators"->{7*512,7*128*512},
"FrameCaps"->{12+256*128,12+256*68,16+256*128*8},
"QuantizedBytes"->(1048576/128)*68+32*12+128,
"QuantizedSavedPercent"->N[100(1-557568/1048704),12],
"LosslessSparseBytes"->547476,"SparseLosslessSavedPercent"->N[100(1-547476/1048704),12],
"HadamardAddsPerBlock"->128*Log2[128],"PayloadBitsPerWeight"->8*68/128|>;
ExportString[result,"RawJSON"]
