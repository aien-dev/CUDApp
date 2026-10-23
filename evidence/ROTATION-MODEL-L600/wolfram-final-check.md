# Independent Wolfram check of the frozen quality result

Tool: `mcp__wolfram__WolframLanguageEvaluator`, `timeConstraint=20`.
Inputs below are the actual stored aggregate means/counts, not invented model data. This checks derived arithmetic and thresholds; it does not independently execute the model.

## Exact evaluator input

```wolfram
rows={{"identity",0.11237303963362714,0.1478824079929612,2446,2258,0.07903087825852005},{"hadamard",0.10472888209302764,0.14854545072709277,2456,2255,-0.05194634711633118},{"zero-aware",0.10376414843971142,0.1459890060650872,2459,2252,-0.06291965974681445},{"nine-level-mutant",0.2153059021445734,0.3028564738534677,2254,2134,0.20453410910800182}}; result=Map[Function[row,<|"arm"->row[[1]],"golden_nll_increase_percent"->100(row[[2]]/0.08896000469688195-1),"heldout_nll_increase_percent"->100(row[[3]]/0.12532170475921625-1),"heldout_agreement_margin_0_1_percent"->N[100 row[[4]]/2521,16],"heldout_agreement_margin_1_percent"->N[100 row[[5]]/2267,16],"heldout_reference_perplexity"->Exp[0.12532170475921625],"heldout_candidate_perplexity"->Exp[row[[3]]],"agpl_mean_nll_change"->row[[6]],"heldout_reject"->(row[[3]]/0.12532170475921625>1.07 || row[[4]]/2521<98/100 || row[[5]]/2267<998/1000)|>],rows];ExportString[<|"scored_positions_per_arm"->(10*512+399),"four_arm_positions"->4(10*512+399),"results"->result|>,"RawJSON"]
```

## Returned JSON (decoded from the `Out[1]` string)

```json
{
	"scored_positions_per_arm":5519,
	"four_arm_positions":22076,
	"results":[
		{
			"arm":"identity",
			"golden_nll_increase_percent":2.6318608026743775e1,
			"heldout_nll_increase_percent":1.8002231358958444e1,
			"heldout_agreement_margin_0_1_percent":97.02499008330028,
			"heldout_agreement_margin_1_percent":99.60299955888840,
			"heldout_reference_perplexity":1.1335130509603708,
			"heldout_candidate_perplexity":1.1593765549307917,
			"agpl_mean_nll_change":7.903087825852005e-2,
			"heldout_reject":true
		},
		{
			"arm":"hadamard",
			"golden_nll_increase_percent":1.772580548964202e1,
			"heldout_nll_increase_percent":1.8531303905015406e1,
			"heldout_agreement_margin_0_1_percent":97.42165807219357,
			"heldout_agreement_margin_1_percent":99.47066607851787,
			"heldout_reference_perplexity":1.1335130509603708,
			"heldout_candidate_perplexity":1.1601455260338407,
			"agpl_mean_nll_change":-5.194634711633118e-2,
			"heldout_reject":true
		},
		{
			"arm":"zero-aware",
			"golden_nll_increase_percent":1.6641347753153113e1,
			"heldout_nll_increase_percent":1.6491398154517235e1,
			"heldout_agreement_margin_0_1_percent":97.54065846886156,
			"heldout_agreement_margin_1_percent":99.33833259814733,
			"heldout_reference_perplexity":1.1335130509603708,
			"heldout_candidate_perplexity":1.1571834659811564,
			"agpl_mean_nll_change":-6.291965974681445e-2,
			"heldout_reject":true
		},
		{
			"arm":"nine-level-mutant",
			"golden_nll_increase_percent":1.420255067186613e2,
			"heldout_nll_increase_percent":1.416632254048518e2,
			"heldout_agreement_margin_0_1_percent":89.40896469654899,
			"heldout_agreement_margin_1_percent":94.13321570357300,
			"heldout_reference_perplexity":1.1335130509603708,
			"heldout_candidate_perplexity":1.3537201562609584,
			"agpl_mean_nll_change":0.20453410910800182,
			"heldout_reject":true
		}
	]
}
```
