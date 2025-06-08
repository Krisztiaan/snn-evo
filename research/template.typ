// BSc Thesis Template - Pázmány Péter Catholic University
// Faculty of Information Technology and Bionics

#let bsc-thesis(
  // The thesis title
  title: [Thesis Title],
  // Author information
  author: (
    name: [Author Name],
    program: [Program Name BSc],
  ),
  // Supervisor information
  supervisor: [Supervisor Name],
  // Year of submission
  year: [2024],
  // Hungarian abstract
  abstract: none,
  // Bibliography file
  bibliography: none,
  // The thesis content
  body,
) = {
  // Set document metadata
  set document(title: title, author: author.name)

  // Set text properties
  set text(
    font: "TeX Gyre Termes",
    size: 11pt,
    lang: "hu",
  )

  // Set page properties
  set page(
    paper: "a4",
    margin: (
      left: 2.5cm,
      right: 2.5cm,
      top: 2.5cm,
      bottom: 2.5cm,
    ),
  )

  // Set paragraph properties
  set par(
    leading: 0.65em * 1.5, // 1.5 line spacing
    justify: true,
  )

  // Configure headings
  set heading(numbering: "1.1")

  // Configure bibliography
  show std.bibliography: set text(10pt)
  show std.bibliography: set block(spacing: 0.5em)
  set std.bibliography(title: text(12pt)[Források], style: "ieee")

  // Code listing counter
  let listing-counter = counter("listing")

  // Show rule for code listings
  show figure.where(kind: "listing"): it => {
    listing-counter.step()
    align(center)[
      #block(
        width: 100%,
        breakable: true,
        [
          #it.body
          #if it.caption != none [
            #v(0.5em)
            #text(weight: "bold")[
              #listing-counter.display().
              #it.supplement
            ]
            #it.caption
          ]
        ],
      )
    ]
  }

  // Title page
  align(center)[
    #grid(
      columns: (auto, 1fr),
      column-gutter: 1em,
      align: (left + horizon),
      image("ITK_logo.jpg", height: 3cm),
      text(size: 14pt)[
        Pázmány Péter Katolikus Egyetem\
        Információs Technológiai és Bionikai Kar
      ],
    )

    #v(2cm)

    #text(size: 16pt)[Önálló laboratórium]

    #v(3cm)

    #text(size: 20pt, weight: "bold")[#title]

    #v(2cm)

    #text(size: 14pt)[
      #author.name\
      #author.program
    ]

    #v(3cm)

    #text(size: 12pt)[
      Témavezető:\
      #supervisor
    ]

    #v(2cm)

    #text(size: 14pt)[#year]
  ]

  pagebreak()

  // Table of contents
  outline(
    title: "Tartalomjegyzék",
    indent: auto,
  )

  pagebreak()

  // Reset page numbering for main content
  set page(numbering: "1")
  counter(page).update(1)

  // Hungarian abstract
  if abstract != none [
    #heading(outlined: false, numbering: none)[Kivonat]
    #abstract
    #pagebreak()
  ]

  // Main content
  body

  bibliography
}

// Appendix helpers
#let appendix-heading(content) = {
  set heading(
    numbering: (..nums) => {
      let vals = nums.pos()
      if vals.len() == 1 {
        return "Függelék " + numbering("A.1", ..nums)
      }
      return numbering("A.1", ..nums)
    },
  )
  counter(heading).update(0)
  heading(content)
}
