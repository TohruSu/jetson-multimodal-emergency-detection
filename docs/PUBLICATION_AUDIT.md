# Source-only publication notes

## Current decision

The independent clean repository has inherited the original project name.
Visibility changes, GitHub Releases and GitHub Profile pinning are separate
owner-authorized actions. SU confirms that SU, SONG, HUO and Martin have each
agreed to MIT licensing of their original contributions, including commercial
use, modification and redistribution. The complete [LICENSE](../LICENSE)
replaces the earlier no-application-license strategy.
[LICENSING.md](LICENSING.md) records team attribution, the MIT grant and
independently applicable third-party rights.

## PyQt and distribution conditions

The team's original application source is MIT-licensed. The GUI continues to
import and subclass PyQt5; its pinned PyPI build remains GPLv3, or an applicable
commercial license may be obtained separately. MIT is GPL-compatible, removing
the previous conflict with the no-general-reuse-license strategy. This does
not waive GPL obligations when distributing a combined application using
GPL PyQt5, nor establish commercial entitlement. The separately installed Qt
runtime has its own LGPLv3 obligations.

The root license does not relicense PyQt, Qt or other third-party components.
A combined GUI, executable, container or bundled environment still needs
review of its actual GPL/commercial and LGPL obligations. No distribution
scenario is legally certified by the source-only audit.
[MODELS.md](MODELS.md) and [LICENSING.md](LICENSING.md) record the dependency
evidence and authoritative licensing references. Model/data redistribution
and privacy checks remain separate.

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
former development records were not imported into this independent history;
their continued retention or availability is not guaranteed. The complete source-tree
comparison, exclusion manifest, Git identity/privacy checks and migration
status were reviewed during private preparation; continued access to separate
private delivery evidence is not guaranteed.

## Independent history and privacy boundary

This repository starts from a file-tree export and a new initial commit; it
does not import development Git objects, branches, tags, PRs or Actions
history. New commit identity and all reachable refs must be verified before
any publication. This independent history does not depend on the continued
availability of the former development repository. It does not erase cached or
independently cloned copies of any earlier public history.

Pattern scans and available refs cannot establish the absence of every
possible secret encoding or inaccessible server record. Visibility changes,
tags, GitHub Releases and GitHub Profile pinning require separate owner
authorization, after validation and publication-condition review.

## Reproducible validation

Local revalidation of the clean export passed all 132 pytest cases with no
skips or expected failures, all three original fusion scenarios, the
16-second five-worker smoke with L1 then L2 and five zero exit codes, the
independent offscreen GUI check with manual reset and worker cleanup, and
`python -m pip check`. [TESTING.md](TESTING.md) records reproducible commands
and coverage. These checks were performed during private preparation;
continued access to separate raw command output is not guaranteed.

The unchanged workflow retains headless Python 3.10 and 3.12 jobs and a
separate Python 3.12 offscreen GUI job, using the pinned action revisions and
`ubuntu-24.04` runner family. This independent repository's initial snapshot
passed its own [hosted run 38285191827](https://github.com/TohruSu/jetson-multimodal-emergency-detection/actions/runs/38285191827):
both headless jobs completed pytest, the original fusion checks and the
five-worker smoke successfully, and the independent offscreen GUI job passed.

The subsequent [hosted run 38331906218](https://github.com/TohruSu/jetson-multimodal-emergency-detection/actions/runs/38331906218)
also succeeded at `91c626be602976b1419ce50f55be1186717dcb30`. Downloaded logs
confirmed 132 passing tests on both Python 3.10 and 3.12, all three original
fusion scenarios and the five-worker smoke with L1 followed by L2 and clean
worker shutdown. The independent offscreen GUI job verified video, depth,
heartbeats, L1/L2 escalation, manual reset, alert logging and worker cleanup.

Each subsequent change must be verified by hosted results for its own commit;
successful runs from the original development repository are separate evidence.

Synthetic, mocked and offscreen checks verify software behavior. They do not
validate Jetson inference accuracy, camera/microphone operation, model
performance, TensorRT conversion or real-world safety. Depth is relative;
speech is not associated with an individual tracked person. Hardware and
end-to-end timing work remain separate from source-only publication. Detailed
technical limitations remain in [ARCHITECTURE.md](ARCHITECTURE.md) and
[SETUP.md](SETUP.md).
