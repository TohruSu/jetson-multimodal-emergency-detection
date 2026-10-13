# Licensing and attribution

## Team MIT authorization

The owner, SU, explicitly confirms that **SU, SONG, HUO, and Martin** have all
agreed to release their respective original code contributions under the
[MIT License](../LICENSE), including commercial use, modification and
redistribution. This supersedes the earlier source-visible, no-application-license
decision. MIT authorization is confirmed; broader reuse permission is no
longer pending.

The complete standard MIT text appears in the root `LICENSE`, with the
collective notice `Copyright (c) 2026 SU, SONG, HUO, and Martin`. Its terms
require preservation of the copyright and permission notices and include the
standard warranty disclaimer. The license covers the team's original work;
it does not relicense independently copyrighted third-party content or external
dependencies. Repository ownership does not make the collaborative application
exclusively SU's work.

[PROVENANCE.md](PROVENANCE.md) records SU's relative-depth, Depth Anything V2,
ROI and pose-to-depth integration; SONG's pose and fall detection; HUO's speech
and help-keyword detection; Martin's fusion, GUI and main integration; and
shared team components. It preserves the original source inventory without
inventing individual commit authorship.

The clean repository has inherited the original project name. Visibility
changes, tags, GitHub Releases and GitHub Profile pinning require separate owner
authorization; MIT adoption does not authorize these actions or model distribution.

## Third-party licensing boundaries

- **PyQt5:** the pinned PyPI build is GPLv3, with an applicable commercial license
  available separately from Riverbank. MIT is GPL-compatible, so the previous
  conflict with withholding general reuse rights no longer applies. This does
  not waive GPL obligations for distribution of a combined application using
  GPL PyQt5. Such distribution must meet the applicable GPL terms, including
  licensing the combined work under the GPL and supplying Corresponding Source
  as required. No commercial entitlement is assumed.
- **Qt runtime:** the separately installed `PyQt5-Qt5` runtime has its own LGPLv3
  terms. Applicable notices, source obligations and the user's ability to
  replace/relink the LGPL library must be considered for an actual distribution.
  PyQt and Qt are distinct components with distinct licensing conditions.
- **NVIDIA asset:** `modules/song/human_pose.json` retains NVIDIA's independently
  applicable MIT copyright and full notice in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The team's root copyright
  notice does not replace NVIDIA's attribution.
- **Other dependencies:** packages and hardware frameworks retain their own
  licenses and notices. The application license does not make the entire
  dependency stack MIT-licensed. [MODELS.md](MODELS.md) records the tested PC
  package terms and the separate hardware/model review boundaries.
- **Excluded artifacts:** MIT authorization for original code grants no
  redistribution rights to absent checkpoints, training data, recordings,
  TensorRT engines or external dependencies. They remain excluded; their
  provenance and applicable rights require separate review.

MIT/GPL compatibility does not certify every packaging or distribution scenario.
Any executable, container, bundled runtime or model distribution must review
its actual contents and applicable obligations. The
[publication audit](PUBLICATION_AUDIT.md) keeps these conditions separate from
source validation and owner authorization. No third-party license or GitHub
Terms of Service is overridden by the root license.

## Authoritative references

- GitHub's [MIT license text](https://choosealicense.com/licenses/mit/) and
  [repository licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
- Riverbank's [PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro)
  and [license documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html#license).
- GNU's [Expat/MIT compatibility guidance](https://www.gnu.org/licenses/license-list.html#Expat)
  and [GPLv3 text](https://www.gnu.org/licenses/gpl-3.0.html).
- Qt's [LGPL obligations](https://www.qt.io/licensing/open-source-lgpl-obligations)
  and the [LGPLv3 text](https://www.gnu.org/licenses/lgpl-3.0.html).
- GitHub's [Terms of Service, section D](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#d-user-generated-content).

These references describe applicable terms and compatibility; they are not a
legal certification of all distributions of this research prototype.
