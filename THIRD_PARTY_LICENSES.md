# Third-party licensing

## Scope of the MIT license

The top-level `LICENSE` applies to original AlphaPulldown code and contributions
for which the project can grant permission. It does not relicense bundled
third-party material, model parameters, databases or other projects, including
AlphaPulldownSnakemake. Existing copyright and attribution notices remain in force.

This inventory records the submodules pinned by AlphaPulldown when the MIT
licensing update was prepared on 2026-11-02. It is not a complete audit of every
dependency or of the software installed in container images.

## Bundled source

The Python distribution includes code from all five submodules below. License
copies in `licenses/` are taken verbatim from the pinned revisions.

| Component | Pinned revision | Evidence and included license |
| --- | --- | --- |
| AlphaFold 2 | `ad317b44c00c2126da9bcbede08ae53f74659823` | [Apache-2.0 upstream](https://github.com/KosinskiLab/alphafold/blob/ad317b44c00c2126da9bcbede08ae53f74659823/LICENSE); [included copy](licenses/AlphaFold2-LICENSE.txt). Preserve per-file DeepMind and other copyright notices. |
| AlphaFold 3 | `98785987139da9781ba271b159c93efaaf23a06f` | [Apache-2.0 upstream](https://github.com/KosinskiLab/alphafold3/blob/98785987139da9781ba271b159c93efaaf23a06f/LICENSE); [included copy](licenses/AlphaFold3-LICENSE.txt). Preserve per-file DeepMind and other copyright notices. |
| ColabFold | `2dca1083d6351944a32aff4f66290a961da744b0` | [MIT upstream](https://github.com/sokrypton/ColabFold/blob/2dca1083d6351944a32aff4f66290a961da744b0/LICENSE), copyright 2021 Sergey Ovchinnikov; [included copy](licenses/ColabFold-LICENSE.txt). |
| AlphaLink2 / Uni-Fold | `44f4bf2aa6b60a8f88ae05fdae12d46d967d5648` | **Scope unresolved.** [README](https://github.com/KosinskiLab/AlphaLink2/blob/44f4bf2aa6b60a8f88ae05fdae12d46d967d5648/README.md) and [setup.py](https://github.com/KosinskiLab/AlphaLink2/blob/44f4bf2aa6b60a8f88ae05fdae12d46d967d5648/setup.py) describe Apache-2.0 code, while [the root license](https://github.com/KosinskiLab/AlphaLink2/blob/44f4bf2aa6b60a8f88ae05fdae12d46d967d5648/LICENSE) contains CC BY 4.0. That text is preserved [here](licenses/AlphaLink2-LICENSE.txt); it does not establish that the AlphaLink2 additions are Apache-2.0. |
| af2plots | `768f35d002ab4380e6ca6ccc6953170b675c7da1` | **Permission unresolved.** No explicit license file or source grant was found at this revision; [setup.py](https://gitlab.com/gchojnowski/af2plots/-/blob/768f35d002ab4380e6ca6ccc6953170b675c7da1/setup.py) sets `license=None`. Public source availability is not a license grant. |

## Parameters and data

- AlphaFold 2 code and parameters have different terms: the code is Apache-2.0,
  while the parameters are CC BY 4.0. See the
  [upstream licensing statement](https://github.com/google-deepmind/alphafold#license-and-disclaimer).
- AlphaFold 3 code is Apache-2.0, but its downloadable parameters and associated
  outputs have [separate terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).
  AlphaPulldown's MIT license does not remove those restrictions.
- AlphaLink2's README declares CC BY 4.0 for model parameters. This declaration
  does not resolve the code-license ambiguity above.
- Sequence/structure databases and container-installed software retain their
  respective terms. They are not covered by AlphaPulldown's MIT grant.

## Distribution metadata and outstanding work

The package deliberately omits a distribution-wide SPDX `license` expression
while permissions for bundled material remain unresolved. `license-files`
includes the project license, this inventory and the upstream license copies in
both wheels and source distributions. An expression such as `MIT` alone would
misdescribe the archive; adding `Apache-2.0` would still leave unresolved material.

Before publishing a release with the revised licensing:

1. Obtain an explicit license grant for af2plots, or replace/remove the bundled
   code and its uses after checking the resulting functionality.
2. Confirm the license of AlphaLink2's additions with its rights holders and
   distinguish code from parameter licensing. Preserve the inherited Uni-Fold
   and AlphaFold notices.
3. Review copied code and data within the bundled components, including notices
   applicable to the actual files shipped; update this inventory if pins change.
4. Set a distribution-wide SPDX expression reflecting the resolved contents and
   verify that all required license texts/notices are present in built archives.

This inventory records open questions; it does not grant missing permissions or
represent the bundled distribution as cleared for unrestricted redistribution.
