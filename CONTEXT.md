# Domain glossary

- **Prediction job**: one independent fold request, including all chains that
  compose that fold and its output directory.
- **Prediction batch**: an ordered set of independent prediction jobs that share
  one homogeneous backend and model configuration.
- **Resident inference**: executing a prediction batch while keeping one set of
  model runners initialized for the lifetime of the batch.
- **Batch manifest**: a JSONL description of prediction jobs. It describes work;
  it does not change fold composition semantics.
- **Job failure**: a recoverable exception associated with one prediction job.
  It is reported after remaining jobs have been attempted.
