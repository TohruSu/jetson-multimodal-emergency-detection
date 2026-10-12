# Publication scope, licensing and attribution

## Owner's selected publication strategy

The owner selected **source-visible publication without a
repository-wide application license**. The intended scope is public viewing
and technical evaluation of the university team's research prototype. No
general open-source grant to reuse, modify, or redistribute the team's
application is being offered. No repository-wide `LICENSE` file is added.

This is the selected strategy, not an outstanding choice of an open-source
license. The absence of MIT or another general reuse license is not, by
itself, a blocker for the proposed source-only publication.

The independent staging repository remains private for review. This decision establishes the
proposed publication scope; it does not authorize a merge, visibility change,
tag, Release, model distribution, or GitHub Profile update.

## Team consent and contribution boundaries

The owner, SU, reports that SU, SONG, HUO, and Martin consented to GitHub
publication of their university team project. This records the owner's report,
not an independently obtained consent agreement. [PROVENANCE.md](PROVENANCE.md)
preserves their responsibilities and the original source inventory. Repository
ownership does not make the collaborative application exclusively SU's work.

Reported publication consent is distinct from an agreement to grant broader
reuse rights. A future general software license would require a separate
agreement covering the relevant contributors' rights, the exact terms, the
covered material, and exclusions. No such agreement is inferred from public
visibility, repository ownership, or the current publication decision.

## Independently applicable rights

The absence of a repository-wide application license does not cancel existing
third-party licenses, notices, copyright exceptions, or rights under GitHub's
Terms of Service. In particular:

- The verified NVIDIA notice applies to the bundled
  `modules/song/human_pose.json` asset only. Its full notice remains in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), independently of the team's
  application publication scope.
- External frameworks and installed packages are separately obtained
  dependencies. Their licenses do not automatically license the team's
  integration code or a particular trained checkpoint.
- GitHub's [Terms of Service, section D](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#d-user-generated-content)
  describe posting responsibilities and rights granted through the service,
  including viewing and forking public repositories. This document does not
  purport to narrow or override those terms or other applicable rights.
- Model checkpoints, training data, recordings, and generated TensorRT
  artifacts remain excluded. Historical checkpoint identity, training-data
  permissions, and redistribution conditions cannot be established from
  filenames or framework licenses. See [MODELS.md](MODELS.md).

Publishing without an application license does not resolve missing permission
to post third-party content or personal information. Any concrete provenance,
permission, or sensitive-history concern must be assessed separately in the
[publication audit](PUBLICATION_AUDIT.md); exclusion of absent artifacts is
not clearance to distribute them later.

The GUI imports and subclasses external PyQt5. The pinned PyPI build is GPL v3,
with a separately offered commercial option; Qt's runtime has distinct LGPL
terms. Compatibility of this GUI integration's source publication with the
selected application terms, or applicable commercial entitlement, remains a
separate review condition. No entitlement is inferred and no team code is
automatically relicensed. [MODELS.md](MODELS.md) records the dependency evidence.

## Source-only publication review checklist

- Preserve SU, SONG, HUO, and Martin's attribution and the original-source
  record; do not imply exclusive ownership of shared integration work.
- Describe the project as source-visible for viewing and evaluation, without
  claiming it is open source or granting a general reuse license.
- Retain independently licensed content's notices and keep unapproved model,
  data, recording, and generated artifacts out of the repository and history.
- Review any separately reported content/provenance concerns before changing
  visibility; require separate owner authorization for publication actions.

Real Jetson validation and a future broader reuse agreement are separate from
the current source-only strategy. Neither is fabricated or silently assumed.

## Guidance consulted

GitHub's [repository licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
states that choosing a license is not mandatory and explains the distinction
between public visibility and an open-source license. Its
[no-license guidance](https://choosealicense.com/no-permission/) also addresses
collaborative authorship and the effect of hosting terms. These are general
references, not a legal determination of this project's ownership or consent.
The official GitHub-maintained source documents were inspected on 2026-10-12:
[repository guidance](https://github.com/github/docs/blob/main/content/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository.md),
[no-license guidance](https://github.com/github/choosealicense.com/blob/gh-pages/no-permission.md),
and [Terms of Service](https://github.com/github/docs/blob/main/content/site-policy/github-terms/github-terms-of-service.md).
