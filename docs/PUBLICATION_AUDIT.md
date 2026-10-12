# Source-only publication notes

## Current decision

**Keep this independent staging repository private. Source-visible publication
is conditional on resolving the PyQt application-terms question and obtaining
separate owner authorization.** The owner has selected public viewing and
technical evaluation without a repository-wide application license; lack of
MIT or another general reuse license is not itself a publication blocker.

The reported consent of SU, SONG, HUO and Martin to GitHub publication is
distinct from any future agreement to grant broader reuse rights.
[LICENSING.md](LICENSING.md) records the selected scope, independently
applicable third-party rights and GitHub's Terms of Service. No application-wide
`LICENSE`, commercial entitlement or automatic relicensing is inferred.

## Separate PyQt publication condition

The GUI imports and subclasses PyQt5. The pinned PyPI PyQt5 5.15.11 build uses
GPL v3 terms; the publisher also offers a separate commercial license. No
determination that publishing this integration source under the selected
application terms is compatible with the applicable PyQt conditions, or
evidence of applicable commercial entitlement, is recorded. Resolve this
before making the repository public. Merely installing PyQt externally does
not establish clearance. The separately installed Qt runtime has distinct
LGPL v3 terms. [MODELS.md](MODELS.md) records the dependency evidence and
distribution boundaries.

This condition is separate from model/data redistribution and from privacy
checks. It does not automatically relicense the team's code. A combined GUI,
executable, container or bundled environment also requires review of its
actual PyQt GPL/commercial and Qt LGPL obligations.

## Preserved content and artifact scope

The independent snapshot preserves the reviewed application implementation,
tests, dependency declarations, workflow and README presentation. All 35
original source paths remain present. Original-input hashes are retained in
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json), and [PROVENANCE.md](PROVENANCE.md)
preserves SU's depth/ROI and pose-to-depth work, SONG's pose/fall work, HUO's
speech/features work, Martin's fusion/UI/main work and shared integration.
The clean initial commit records a portfolio snapshot, not individual
authorship of the historical university team project.

The 510-byte NVIDIA skeleton remains unchanged, with SHA-256
`86704ad74020e1c329bd1c73f41da37f98e7728948a1285375a437756bae1bcd`.
Its scoped notice is preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); it does not license the team's
application or model weights.

No model checkpoints, training data, recordings, generated TensorRT engines,
vendored runtime packages or environment caches belong in the publication
snapshot. Missing historical model identity, training-data consent and
redistribution rights remain unverified. Filenames and framework licenses
do not establish those rights. No model is downloaded or deserialized for
synthetic validation. Any later artifact distribution requires its own
provenance and permission review.

Two internal phase reports are excluded from this portfolio snapshot after
checking that retained documentation no longer depends on them. Private
development SHAs, PR links and Actions-history links are omitted. The
original repository retains its historical records. The complete source-tree
comparison, exclusion manifest, Git identity/privacy checks and migration
status are recorded separately in the private delivery evidence.

## Independent history and privacy boundary

This repository starts from a file-tree export and a new initial commit; it
does not import development Git objects, branches, tags, PRs or Actions
history. New commit identity and all reachable refs must be verified before
any publication. Independent staging preserves the original repository rather
than rewriting or deleting its records. It does not erase cached or
independently cloned copies of any earlier public history.

Pattern scans and available refs cannot establish the absence of every
possible secret encoding or inaccessible server record. Public visibility,
repository renaming, release creation and Profile changes remain separately
authorized actions, after validation and publication-condition review.

## Reproducible validation

Local revalidation of the clean export passed all 132 pytest cases with no
skips or expected failures, all three original fusion scenarios, the
16-second five-worker smoke with L1 then L2 and five zero exit codes, the
independent offscreen GUI check with manual reset and worker cleanup, and
`python -m pip check`. [TESTING.md](TESTING.md) records reproducible commands
and coverage. Exact command output is retained outside the publication tree
in the private migration evidence.

The unchanged workflow retains headless Python 3.10 and 3.12 jobs and a
separate Python 3.12 offscreen GUI job, using the pinned action revisions and
`ubuntu-24.04` runner family. Hosted results must correspond to the independent
repository's clean commit. Successful runs from the original repository do
not establish that new staging CI has passed; hosted validation remains
required after a private staging repository is created and pushed.

Synthetic, mocked and offscreen checks verify software behavior. They do not
validate Jetson inference accuracy, camera/microphone operation, model
performance, TensorRT conversion or real-world safety. Depth is relative;
speech is not associated with an individual tracked person. Hardware and
end-to-end timing work remain separate from source-only publication. Detailed
technical limitations remain in [ARCHITECTURE.md](ARCHITECTURE.md) and
[SETUP.md](SETUP.md).
