# Synthetic NLI Review Sample

- Input file: `E:\AI\NLP-Text-Classification-Model-Analysis\data\generated\targeted_nli_run4.json`
- Sampling seed: `13`
- Target examples per category: `12`
- Total sampled examples: `48`

## long-premise short-hypothesis reasoning

- Sampled examples: `12`
- Label mix: entailment: 4, contradiction: 4, neutral: 4

### Example 1
- Category: long-premise short-hypothesis reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Felix drafted the visit log before noon. Daniel checked the figures but did not edit the visit log. After the corrections were approved, Felix emailed the final visit log to the director.
- Hypothesis: Felix emailed the final visit log.

### Example 2
- Category: long-premise short-hypothesis reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: Hugo drafted the visit log before noon. Rosa checked the figures but did not edit the visit log. After the corrections were approved, Hugo emailed the final visit log to the director.
- Hypothesis: Rosa emailed the final visit log.

### Example 3
- Category: long-premise short-hypothesis reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the east lab: Omar, Zane, and Hugo. Omar had the entry key, Zane had the visitor pass, and Hugo had both. The coordinator explained that entry required both documents. In the end, only Hugo entered the east lab.
- Hypothesis: Zane requested a temporary pass later.

### Example 4
- Category: long-premise short-hypothesis reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Mina drafted the site memo before noon. Clara checked the figures but did not edit the site memo. After the corrections were approved, Mina emailed the final site memo to the director.
- Hypothesis: Mina emailed the final site memo.

### Example 5
- Category: long-premise short-hypothesis reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the records room: Victor, Felix, and Yara. Victor had the code card, Felix had the visitor pass, and Yara had both. The coordinator explained that entry required both documents. In the end, only Yara entered the records room.
- Hypothesis: Victor entered the records room.

### Example 6
- Category: long-premise short-hypothesis reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The backup pump failed at 08:45. Victor shut off the line, and Theo replaced the clogged part. Once the pressure returned to normal, Victor restarted the backup pump at 11:00. The supervisor wrote the incident report after the restart.
- Hypothesis: A replacement machine arrived that afternoon.

### Example 7
- Category: long-premise short-hypothesis reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the records room: Clara, Lena, and Nora. Clara had the entry key, Lena had the safety form, and Nora had both. The coordinator explained that entry required both documents. In the end, only Nora entered the records room.
- Hypothesis: Nora entered the records room.

### Example 8
- Category: long-premise short-hypothesis reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the records room: Yara, Felix, and Grace. Yara had the code card, Felix had the visitor pass, and Grace had both. The coordinator explained that entry required both documents. In the end, only Grace entered the records room.
- Hypothesis: Yara entered the records room.

### Example 9
- Category: long-premise short-hypothesis reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The backup pump failed at 06:00. Victor shut off the line, and Lena replaced the clogged part. Once the pressure returned to normal, Victor restarted the backup pump at 08:15. The supervisor wrote the incident report after the restart.
- Hypothesis: A replacement machine arrived that afternoon.

### Example 10
- Category: long-premise short-hypothesis reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the testing room: Iris, Ben, and Ava. Iris had the code card, Ben had the clearance note, and Ava had both. The coordinator explained that entry required both documents. In the end, only Ava entered the testing room.
- Hypothesis: Ava entered the testing room.

### Example 11
- Category: long-premise short-hypothesis reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: Three visitors checked in for the records room: Rosa, Grace, and Mina. Rosa had the code card, Grace had the clearance note, and Mina had both. The coordinator explained that entry required both documents. In the end, only Mina entered the records room.
- Hypothesis: Rosa entered the records room.

### Example 12
- Category: long-premise short-hypothesis reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: During the morning audit, the team counted 10 sealed crates on the front pallet and 2 damaged ones near the door. Only the sealed crates were moved to the packing desk; the damaged ones stayed by the side shelf for inspection. Before lunch, the supervisor signed the transfer sheet for the crates that went to the packing desk.
- Hypothesis: The sealed crates were shipped to another building.

## negation

- Sampled examples: `12`
- Label mix: entailment: 4, contradiction: 4, neutral: 4

### Example 1
- Category: negation
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The maintenance log for the library annex says the delivery gate was not working.
- Hypothesis: The delivery gate was not working.

### Example 2
- Category: negation
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The audit found that none of the 7 gates on the service hall were open.
- Hypothesis: At least one gate on the service hall was open.

### Example 3
- Category: negation
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: None of the 10 gates on the west wall were open.
- Hypothesis: The gates on the west wall were repainted last month.

### Example 4
- Category: negation
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Ava did not say that the backup server was wet; Ava said that the label printer was broken.
- Hypothesis: Ava said the label printer was broken.

### Example 5
- Category: negation
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The transcript shows that Yara did not say the east cabinet was wet; the recorded claim was that the label printer was empty.
- Hypothesis: Yara said the east cabinet was wet.

### Example 6
- Category: negation
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: During the closing check, the technician noted that the display screen was not locked in the records office.
- Hypothesis: The display screen was replaced after the inspection.

### Example 7
- Category: negation
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: In the briefing, Priya did not claim the delivery gate was empty. Instead, Priya said the ticket scanner was jammed.
- Hypothesis: Priya said the ticket scanner was jammed.

### Example 8
- Category: negation
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: According to the inspection note, the display screen was not active when the supervisor reviewed the repair bay.
- Hypothesis: The display screen was active.

### Example 9
- Category: negation
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: Lena did not say that the heater was offline; Lena said that the camera was unlocked.
- Hypothesis: Lena repaired the heater.

### Example 10
- Category: negation
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: Not all 4 invoices were complete; 1 still needed signatures.
- Hypothesis: Some invoices were not complete.

### Example 11
- Category: negation
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The audit found that none of the 9 cabinets on the storage row were open.
- Hypothesis: At least one cabinet on the storage row was open.

### Example 12
- Category: negation
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: In the briefing, Ava did not claim the ticket scanner was missing. Instead, Ava said the storage cabinet was unlocked.
- Hypothesis: Ava repaired the ticket scanner.

## numeric contradiction

- Sampled examples: `12`
- Label mix: entailment: 4, contradiction: 4, neutral: 4

### Example 1
- Category: numeric contradiction
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: On the count sheet, the north shelf had 8 sample boxes while the south shelf had 7.
- Hypothesis: There were 15 sample boxes altogether.

### Example 2
- Category: numeric contradiction
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The field report recorded exactly 12 foxes near the riverbank checkpoint.
- Hypothesis: At least 17 foxes were near the riverbank checkpoint.

### Example 3
- Category: numeric contradiction
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: During the count, the team found exactly 14 red folders on the side shelf.
- Hypothesis: Two of the red folders were opened that evening.

### Example 4
- Category: numeric contradiction
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: On the count sheet, the front table had 8 tickets while the north shelf had 7.
- Hypothesis: There were 15 tickets altogether.

### Example 5
- Category: numeric contradiction
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The field report recorded exactly 21 seals near the south field checkpoint.
- Hypothesis: At most 18 seals were near the south field checkpoint.

### Example 6
- Category: numeric contradiction
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: During the count, the team found exactly 23 yellow folders on the east table.
- Hypothesis: Two of the yellow folders were opened that evening.

### Example 7
- Category: numeric contradiction
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The side table held 4 sample boxes, and the north shelf held 6 sample boxes.
- Hypothesis: There were 10 sample boxes altogether.

### Example 8
- Category: numeric contradiction
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: In the final count, the north rack had 12 manuals and the south rack had 8.
- Hypothesis: The south rack stored more manuals than the north rack.

### Example 9
- Category: numeric contradiction
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The west room stored 8 samples, while the east room stored 14.
- Hypothesis: Both areas were cleaned after the count.

### Example 10
- Category: numeric contradiction
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: During the survey, the team counted exactly 12 cranes near the south field checkpoint.
- Hypothesis: At least 9 cranes were near the south field checkpoint.

### Example 11
- Category: numeric contradiction
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The back shelf stored 17 forms, while the north rack stored 7.
- Hypothesis: The north rack stored more forms than the back shelf.

### Example 12
- Category: numeric contradiction
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: During the survey, the team counted exactly 21 herons near the east marsh checkpoint.
- Hypothesis: The herons crossed the checkpoint before sunrise.

## temporal/date reasoning

- Sampled examples: `12`
- Label mix: entailment: 4, contradiction: 4, neutral: 4

### Example 1
- Category: temporal/date reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The final review happened on 2024-01-24. The training test happened later, on 2024-01-27.
- Hypothesis: The training test happened later than the final review.

### Example 2
- Category: temporal/date reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: Theo submitted the draft at 11:15 on 2021-06-10. The editor approved it at 13:30. The print team started at 14:45.
- Hypothesis: The print team started before the draft was approved.

### Example 3
- Category: temporal/date reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The plan says the trial run lasted 4 days, beginning on 2024-08-19 and ending on 2024-08-22.
- Hypothesis: The trial run switched to an online format on the final day.

### Example 4
- Category: temporal/date reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The timeline for 2022-11-23 lists the draft submission at 12:50, approval at 15:20, and printing at 16:50.
- Hypothesis: The draft was approved after it was submitted.

### Example 5
- Category: temporal/date reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The training session happened on 2024-06-30. The trial run happened on 2024-07-01.
- Hypothesis: The training session happened after the trial run.

### Example 6
- Category: temporal/date reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The follow-up call happened on 2022-07-13. The training test happened later, on 2022-07-18.
- Hypothesis: The training test lasted two hours.

### Example 7
- Category: temporal/date reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The inspection started at 10:20 on 2021-10-08. The review call started at 13:50 on the same day.
- Hypothesis: The inspection and the review call happened on the same day.

### Example 8
- Category: temporal/date reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The briefing ran for 2 days, from 2021-05-22 through 2021-05-23.
- Hypothesis: The briefing lasted 1 day.

### Example 9
- Category: temporal/date reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The schedule for 2021-08-27 shows the safety drill at 14:40 and the debrief at 17:00.
- Hypothesis: The debrief lasted thirty minutes.

### Example 10
- Category: temporal/date reasoning
- Label: entailment
- Generation method: diverse_targeted_nli_v2
- Premise: The log shows the site visit on 2022-07-05 and the follow-up call on 2022-07-08.
- Hypothesis: The follow-up call happened later than the site visit.

### Example 11
- Category: temporal/date reasoning
- Label: contradiction
- Generation method: diverse_targeted_nli_v2
- Premise: The log shows the site visit on 2023-06-21 and the training test on 2023-06-27.
- Hypothesis: The site visit and the training test happened on the same day.

### Example 12
- Category: temporal/date reasoning
- Label: neutral
- Generation method: diverse_targeted_nli_v2
- Premise: The training test happened on 2022-07-27. The follow-up call happened later, on 2022-08-01.
- Hypothesis: The follow-up call lasted two hours.
