## CLOZE_MATH

Example:
![](./CLOZE_MATH.png)

Task:
Simplify as much as possible: \\\\frac{1+\\\\frac{1}{a}}{a+\\\\frac{1}{a}}=\\\\frac{[--]a^{2}+[--]a+[--]}{[--]a^{2}+[--]a+[--]}

Answer:
[
    "0",
    "1",
    "1",
    "1",
    "0",
    "1"
]

Explanation: every -- in input is replaced by the corresponding answer  
Easily applicable to answer prediction ✅

## CLOZE_TEXT

Example:
<!-- ![](./CLOZE_TEXT.png) -->

Task:
<p>Die blauen Fr&uuml;hlings<strong>augen &nbsp; &nbsp; &nbsp; &nbsp;<img data-replacement-cloze-text-input=\\"0\\" data-gap-data=\\"&quot;Metapher&quot;\\" class=\\"fr-fic fr-dii\\"></strong><br><strong>Schaun</strong> aus dem Gras hervor; &nbsp;&nbsp;<img data-replacement-cloze-text-input=\\"1\\" data-gap-data=\\"&quot;Personifikation&quot;\\" class=\\"fr-fic fr-dii\\"><br>Das sind die lieben Veilchen,<br>Die ich zum Strau&szlig; erkor.<br><br></p><p>Ich pfl&uuml;cke sie und denke,<br>Und die Gedanken all,<br>Die mir im Herzen seufzen,<br>Singt laut die <strong>Nachtigall</strong>. &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<img data-replacement-cloze-text-input=\\"2\\" data-gap-data=\\"&quot;Symbol&quot;\\" class=\\"fr-fic fr-dii\\"><br><br></p><p>Ja, was ich denke, singt sie<br>Lautschmetternd, dass es schallt;<br><strong>Mein z&auml;rtliches Geheimnis</strong><br><strong>Wei&szlig; schon der ganze Wald</strong>. &nbsp;&nbsp;<img data-replacement-cloze-text-input=\\"3\\" data-gap-data=\\"&quot;Ironie&quot;\\" class=\\"fr-fic fr-dii\\"></p>

Answer:
[
    "Metapher", 
    "Personifikation", 
    "Symbol", 
    "Ironie"
], 

Explanation: every <img data-replacement-cloze-text-input=\\"answer_index\\" data-gap-data=\\"&quot; text_to_guess &quot;\\" class=\\"fr-fic fr-dii\\"> in input is replaced by the corresponding answer  
Easily applicable to answer prediction ✅

## CLOZE_TEXT_DROPDOWN

Example:
<!-- ![](./CLOZE_TEXT_DROPDOWN.png) -->

Task:
Die Gleichung $2mx+3m=6x+9$ hat &nbsp;<img data-replacement-cloze-text-dropdown=\\"0\\" data-gap-data=\\"{&quot;indexOfAnswer&quot;:0,&quot;options&quot;:[&quot;für alle Werte von m&quot;,&quot;nur für m = 3&quot;,&quot;falls $m\\\\\\\\neq$0&quot;]}\\" class=\\"fr-fic fr-dii\\">&nbsp;mindestens eine Lösung. Falls <img data-replacement-cloze-text-dropdown=\\"1\\" data-gap-data=\\"{&quot;indexOfAnswer&quot;:1,&quot;options&quot;:[&quot;m = 3&quot;,&quot;m $\\\\\\\\neq$ 3&quot;]}\\" class=\\"fr-fic fr-dii\\">, dann ist die einzige Lösung $x=-\\\\frac{3}{2}$.

Answer:
[
    "für alle Werte von $m$",
    "$m=3$"
]

Explanation: every <img data-replacement-cloze-text-input=\\"answer_index\\" data-gap-data=\\"&quot; text_to_guess &quot;\\" class=\\"fr-fic fr-dii\\"> in input is replaced by the corresponding answer  
Easily applicable to answer prediction ✅

## DND_GROUP

Example:
<!-- ![](./DND_GROUP.png) -->

Task: Associate words to group
    "dragAndDrop": [
        {
            "answers": [
                "grübeln",
                "hirnen"
            ],
            "question": "<p>Umgangssprache (z.B. im Alltag verwendet, gesprochen)</p>"
        },
        {
            "answers": [
                "denken"
            ],
            "question": "<p>Standardsprache (z.B. in der Zeitung)</p>"
        },
        {
            "answers": [
                "sinnen"
            ],
            "question": "<p>Gehobene Sprache (z.B. in Kirche, vor Gericht, in Poesie)</p>"
        }
    ],

Answer:
"groups": [
    [
        "sexy",
        "antörnend"
    ],
    [
        "attraktiv",
        "fesch"
    ],
    [
        "berückend"
    ]
]  
Easily applicable to answer prediction ❌

## DND_IN_TEXT

Example:
<!-- ![](./DND_IN_TEXT.png) -->

Task:
<p><img data-replacement-dnd-in-text=\\"0\\" data-gap-data=\\"{&quot;text&quot;:&quot;Heute&quot;,&quot;answer&quot;:&quot;Satzanfang&quot;}\\" class=\\"fr-fic fr-dii\\">&nbsp;hielten wir <img data-replacement-dnd-in-text=\\"1\\" data-gap-data=\\"{&quot;text&quot;:&quot;fest&quot;,&quot;answer&quot;:&quot;Verbzusatz zu festhalten&quot;}\\" class=\\"fr-fic fr-dii\\">: <img data-replacement-dnd-in-text=\\"2\\" data-gap-data=\\"{&quot;text&quot;:&quot;Morgen&quot;,&quot;answer&quot;:&quot;Satzanfang nach Doppelpunkt&quot;}\\" class=\\"fr-fic fr-dii\\">&nbsp;werden wir <img data-replacement-dnd-in-text=\\"3\\" data-gap-data=\\"{&quot;text&quot;:&quot;aufhören&quot;,&quot;answer&quot;:&quot;Verb&quot;}\\" class=\\"fr-fic fr-dii\\"><img data-replacement-dnd-in-text=\\"4\\" data-gap-data=\\"{&quot;text&quot;:&quot;unordentlich&quot;,&quot;answer&quot;:&quot;Adjektiv&quot;}\\" class=\\"fr-fic fr-dii\\">&nbsp;zu sein, wenigstens in unserem <img data-replacement-dnd-in-text=\\"5\\" data-gap-data=\\"{&quot;text&quot;:&quot;&quot;,&quot;answer&quot;:&quot;&quot;}\\" class=\\"fr-fic fr-dii\\">.</p>

Answer:
[
    "Satzanfang",
    "Adjektiv",
    "Satzanfang nach Doppelpunkt",
    "Verb",
    "Verbzusatz zu festhalten",
    "Nomen"
]

Explanation: every <img data-replacement-cloze-text-input=\\"answer_index\\" data-gap-data=\\"&quot; text_to_guess &quot;\\" class=\\"fr-fic fr-dii\\"> in input is replaced by the corresponding answer  
Easily applicable to answer prediction ✅

## DND_ORDER

Example:
<!-- ![](./DND_ORDER.png) -->

Task:
"description": "$P$ ist ein beliebiger Punkt im Innern des Dreiecks $ABC$. Zeige, dass $RS\\\\|BC$ gilt.<br><img src=\\"https://files.taskbase.com/api/file/80Vgem9dmXaatKBvGgcwEf\\"><br>Die Lösung der Aufgabe ist durcheinander geraten. Bringe die einzelnen Schritte in die korrekte Reihenfolge.",
"dragAndDrop": ["$\\\\overline{AB}\\\\div \\\\overline{AR}=\\\\overline{AP}\\\\div \\\\overline{AQ}$ (1. Strahlensatz)", "$\\\\overline{AC}\\\\div \\\\overline{AS}=\\\\overline{AP}\\\\div \\\\overline{AQ}$ (ebenso)", "$\\\\overline{AB}\\\\div \\\\overline{AR}=\\\\overline{AC}\\\\div \\\\overline{AS}$", "$RS\\\\|BC$ (Umkehrung 1. Strahlensatz)"], 

Answer:
[
    "$RS\\\\|BC$ (Umkehrung 1. Strahlensatz)",
    "$\\\\overline{AB}\\\\div \\\\overline{AR}=\\\\overline{AC}\\\\div \\\\overline{AS}$",
    "$\\\\overline{AC}\\\\div \\\\overline{AS}=\\\\overline{AP}\\\\div \\\\overline{AQ}$ (ebenso)",
    "$\\\\overline{AB}\\\\div \\\\overline{AR}=\\\\overline{AP}\\\\div \\\\overline{AQ}$ (1. Strahlensatz)"
]


Easily applicable to answer prediction ❌

## DND_PAIRS

Example:
<!-- ![](./DND_PAIRS.png) -->

Task:
[
    {
        "answer": "<p>Bern hat noch nicht geantwortet.</p>", 
        "question": "<p><strong>Metonymie:</strong> Ersetzung eines gebr&auml;uchlichen Wortes durch ein anderes, das zu ihm in unmittelbarer Beziehung steht</p>"
    }, 
    {
        "answer": "<p>Iss deinen Teller auf!</p>", 
        "question": "<div title=\\"Page 1\\"><p><strong>Synekdoche:</strong> Ersetzung eines Ausdrucks, wobei entweder ein Teil f&uuml;r das Ganze oder das Ganze f&uuml;r einen Teil steht.</p></div>"
    }, 
    {
        "answer": "<p>Als sie davon erfuhr, ist sie aus allen Wolken gefallen.</p>", 
        "question": "<p><strong>Metapher:</strong> Einem Wort wird ein bildhafter Ausdruck zugeordnet, der aus einem vollkommen unabh&auml;ngigen Bedeutungszusammenhnag stammt.&nbsp;</p>"
    }, 
    {
        "answer": "<div title=\\"Page 6\\"><p>Mir schlug das Herz; geschwind zu Pferde/ Und fort, wild, wie ein Held zur Schlacht! (Goethe)</p></div>", 
        "question": "<p><strong>Vergleich:</strong> Gegen&uuml;berstellung zweier Sachverhalte, die eine Gemeinsamkeit haben. Verbunden werden sie durch Vergleichspartikel (wie, gleich, als ob...).</p>"
    }
]

Answer:
[
    "<p>Ich gehe in die Bibliothek, um das Buch zu verlängern.</p>",
    "<p>Iss deinen Teller auf!</p>",
    "<p>Als sie davon erfuhr, ist sie aus allen Wolken gefallen.</p>",
    "<div title=\\"Page 6\\"><p>Mir schlug das Herz; geschwind zu Pferde/ Und fort, wild, wie ein Held zur Schlacht! (Goethe)</p></div>"
]

Easily applicable to answer prediction ❓


## FIX_TEXT

Example:
<!-- ![](./FIX_TEXT.png) -->

Task:
"description": "<strong>Korrigiere in den folgenden Sätzen die fehlerhafte Gross- und Kleinschreibung.&nbsp;</strong>",
"textWithMistakes": {
    "solution": "Trotz all dem Weh und Ach meiner Eltern: Meine Schuld ist es nicht, wahrscheinlich ist gar niemand schuld.\\n\\nImmer wieder dieses Hin und Her, dieses Gezeter und Gestreite! Immer wieder dieses elende Entweder-oder möchte ich mich endlich wieder frei entscheiden können zwischen allen Optionen.\\n\\nHeute hielten wir fest, dass wir morgen aufhören werden, unordentlich zu sein, wenigstens in unserem Spint. Unsere innere Stimme hat deutlich Ja gesagt!",
    "stimulus": "Trotz all dem weh und ach meiner Eltern: Meine schuld war es nicht, wahrscheinlich ist gar niemand Schuld.\\n\\nImmer wieder dieses hin und her, dieses gezeter und gestreite! Immer wieder dieses elende entweder-oder, ich möchte mich endlich wieder frei entscheiden können zwischen allen Optionen.\\n\\nHeute hielten wir Fest, dass wir Morgen aufhören werden, Unordentlich zu sein, wenigstens in unserem Spint. Unsere innere Stimme hat deutlich ja gesagt!"
}

Answer:
"fixedText": "Heute hielten wir fest, dass wir Morgen aufhören werden, Unordentlich zu sein, wenigstens in unserem Spind. Unsere innere Stimme hat deutlich ja gesagt!"

Explanation: every -- in input is replaced by the corresponding answer
Easily applicable to answer prediction ✅

## HIGHLIGHT

Example:
<!-- ![](./HIGHLIGHT.png) -->

Task:
<p><strong>Der nachfolgende Text ist jeweils durch einen Schr&auml;gstrich in drei Abschnitte gegliedert. Markiere in jedem Abschnitt eine These (Behauptung, Forderung/Empfehlung, Werturteil).</strong></p>
{
    "token": "wurden.",
    "isBold": false,
    "correct": false
} for each word in text

Answer: [false, false, true, true, false, false, ...]

Explanation: highlighted state of each word in text

Easily applicable to answer prediction ✅

## MATH_STEP_BY_STEP

Example:
<!-- ![](./MATH_STEP_BY_STEP.png) -->

Task:
"description": "Es gilt $AE \\\\| BD$. Berechne $x$.<br><img src=\\"https://files.taskbase.com/api/file/86GmCb1qHYJ8c07ER6ZGNt\\" style=\\"width: 300px;\\" class=\\"fr-fic fr-dib\\">", 
"mathProblem": "(x+0.5)/1.2=0.5/0.4", 

Answer: list of steps
[
    "\\\\frac{(x+0.5)}{1.2}=\\\\frac{0.5}{0.4}",
    ""
]


Easily applicable to answer prediction ✅ (we must exclude image-based problems)

## MULTIPLE_CHOICE

Example:
<!-- ![](./MULTIPLE_CHOICE.png) -->

Task:
"description": "Welche ist die Lösungsmenge der Ungleichung<br>$\\\\left(x-3\\\\right)^2\\\\ge0?$",
"choices": [
        {
            "content": "$\\\\mathbb{L}=[3,\\\\infty]$",
            "correct": false
        },
        {
            "content": "$\\\\mathbb{L}=\\\\mathbb{R}$",
            "correct": true
        },
        {
            "content": "$\\\\mathbb{L}=]-\\\\infty,3]$",
            "correct": false
        },
        {
            "content": "Es gibt keine Lösung.",
            "correct": false
        }
    ],

Answer:
"selections": {"2": true}

Easily applicable to answer prediction ✅

## MULTI_COLOR_HIGHLIGHT

Example:
<!-- ![](./MULTI_COLOR_HIGHLIGHT.png) -->

Task:
<p>Markiere die Tropen und Figuren im Gedichtauszug &quot;Willkommen und Abschied&quot; von Johann Wolfgang Goethe.</p>
{
    "label": "Anapher",
    "token": "Es",
    "isBold": false
},
{
    "token": "schlug",
    "isBold": false
},
{
    "token": "mein",
    "isBold": false
},
{
    "label": "Symbol",
    "token": "Herz,",
    "isBold": false
}, ...

Answer: {"label": "Anapher", "token": "Es", "isBold": false}, {"label": "", "token": "schlug", "isBold": false}, ...

Easily applicable to answer prediction (not the easiest one)

## OPEN_TASK

Example:
<!-- ![](./OPEN_TASK.png) -->

Task:
<strong>Beschreibe im unten stehenden Auszug aus Ogis Neujahrsrede die Wirkung der markierten Stilmittel.</strong><br><br><p>Wir danken den drei Generationen, die unserem Land im letzten Jahrhundert gedient haben; die es mit ihrem <strong>Wissen, Können und Wollen</strong> durch turbulente Zeiten führten; die ein <strong>modernes, gerechtes und soziales</strong> Staatswesen geschaffen haben. Es bietet uns allen <strong>Sicherheit, Stabilität und Wohlstand</strong>. Dafür sind wir dankbar.&nbsp;</p><p>Antworten&nbsp;</p>

Answer:
Open text

Easily applicable to answer prediction ✅✅

## SEPARATE_TEXT

Example:
<!-- ![](./SEPARATE_TEXT.png) -->

Task:
(in this task there was no description, probably if we use it we should create a sort of prompt)
{
    "text": "österreichisch",
    "separatorRight": "NONE"
},
{
    "text": "Beistrich",
    "separatorRight": "COMMA"
}, ...

Answer:
{
    "text": "Selbstverständlich",
    "separatorRight": "NONE"
},
{
    "text": "würde",
    "separatorRight": "NONE"
},
{
    "text": "es",
    "separatorRight": "COMMA"
},


Easily applicable to answer prediction ❓

## SOLUTION_FIELD

Example:
<!-- ![](./SOLUTION_FIELD.png) -->

Task:
Löse die folgende Gleichung (ohne Fallunterscheidung) nach $x$ auf:<br>$ax+b=9b$

Answer:
\\\\frac{8b}{a}

Easily applicable to answer prediction ✅✅
