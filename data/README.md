# Data

The data are not included in this repository. They were obtained from the TriNetX federated health research network under a data use agreement that does not allow sharing. Researchers can apply for access through their institution via the TriNetX platform (https://trinetx.com).

The scripts expect one Excel file at `data/merged_90_c1.xlsx`, or at the path set by the `BGCS_DATA` environment variable. It has one row per patient and 161 columns.

- `patient_id`, an identifier that is dropped before modelling
- `dialysis`, the outcome label (1 = initiated dialysis, 0 = did not), as defined in the paper
- 159 binary (0/1) features named `code_<code>` that flag diagnosis, procedure and medication codes, for example `code_G0480` or `code_Z09`

The cohort in the paper has 4,328 patients, 241 of whom initiated dialysis.

Everything in this folder except this README is ignored by git.
