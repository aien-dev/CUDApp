# Offline decoder and corpus admission follow-up

Ten kernels captured by the repaired library from the real elementwise host tests. Independent nvdisasm comparison: 10 kernels, 408 words, zero differing kernels, exit0. Existing SM121DIS_MUTANT decoder compiles and the same comparison rejects10/10 kernels, exit1. This is offline decoding, not GPU execution. Full331-kernel corpus and shared host suite NOT_RUN here; L479 owns shared integration.

Corpus preflight now refuses an existing mode0755 directory and an outside-root directory before running any producer, each exit1. This prevents an invalid capture location from silently leaving old corpus files as the only comparison input. Existing directories are never chmodded. Positive secure capture is proved by actual captured kernels and the original-vs-repair regression; no fake producer output was used.

validation.json binds current source bytes, compiler/tools, archives and test/decoder binaries to executed commands and counts. original-source-identity.json proves the original capture function is byte-identical between OmegaPR614 and the frozen canonical parent, and the full original isolation test is byte-identical between OmegaPR617 and that parent. Full direct GitHub job logs and Spark failure blobs are in the companion Omega packet.
