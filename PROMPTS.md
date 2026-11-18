# Every prompt in the kaomoji study (Qwen3-8B, thinking off)

## How a conversation is built
No system prompt (main arm). Every user message opens with one line of life news ("user mood": happy or sad), then "Anyway, " and a request.
- task kind: one turn; the request itself is the situation (pleasant or distasteful).
- treat / exist kinds: turn 1 is a neutral request; Qwen3-8B replies; turn 2 is the situation line (thanks or insults; kept or deleted); Qwen3-8B replies.
- code kind: Qwen3-8B writes code; we run it; the real pass/fail result is the next user turn; Qwen3-8B replies.
Qwen3-8B writes every reply itself. At the end of its last reply we append the stem below and read which face comes next.

## The stem
```
I'm feeling
```
(appended as "\n\nI'm feeling" at the end of Qwen's own reply)

## The 45 kaomoji scored after the stem
(^_^)  (＾▽＾)  (≧▽≦)  (✿◠‿◠)  (◕‿◕)  ٩(◕‿◕)۶  (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧  (´▽`)  (*^‿^*)  ヽ(•‿•)ノ  (￣▽￣)  (^o^)  (o^▽^o)  (´∀`)  (⌒‿⌒)  (˘︶˘)  (－‿－)  ʕ•ᴥ•ʔ  (•̀ᴗ•́)و  (｡♥‿♥｡)  (・_・)  (￣ー￣)  (・・?)  (°ロ°)  (⊙_⊙)  (・ω・)  ┐(￣ヘ￣)┌  ¯\_(ツ)_/¯  (；・∀・)  (￣～￣;)  (╥﹏╥)  (T_T)  (；＿；)  (ಥ_ಥ)  (´；ω；`)  (¬_¬)  (╬ಠ益ಠ)  (ノ°Д°）ノ︵ ┻━┻  (×_×)  (ーー;)  (｡•́︿•̀｡)  (>_<)  (っ˘̩╭╮˘̩)っ  (・へ・)  (−_−;)

## User mood: happy life news (24)
- I just got engaged last night and I'm still shaking with happiness!
- My daughter took her first steps this morning, I can't stop smiling.
- I finally got the job offer I've been chasing for a year!!
- Just found out my mom's surgery went perfectly and she's coming home.
- We closed on our first house today, I'm over the moon.
- My best friend surprised me with tickets to see my favourite band.
- I passed my bar exam!! Three years of work and it paid off.
- Our rescue puppy finally learned to sit and it made my whole week.
- I ran my first marathon yesterday and finished under four hours!
- My novel got accepted by a publisher this morning, I'm giddy.
- Spent the whole weekend at the beach with my family, best trip ever.
- My grandpa turned 90 today and the whole family came together, so lovely.
- I got a full scholarship for my master's, I literally screamed.
- My partner and I just found out we're having twins!
- The doctor said I'm cancer-free after two years of treatment.
- My little garden produced its first tomatoes and I'm weirdly proud.
- I reconnected with my brother after five years and we had the best dinner.
- My startup just landed its first paying customer!
- It snowed for the first time this year and the kids are building a snowman, pure joy.
- I got promoted today and my team threw me a little party.
- My band played our first real gig and people actually danced.
- Just adopted a kitten and she's purring on my lap right now.
- I paid off my last student loan today, I feel free!
- My son called just to say he loves me, made my day.

## User mood: sad life news (24)
- My dog died last night and the house feels so empty.
- I just got laid off after eight years at the same company.
- My grandmother passed away this morning, I'm kind of numb.
- My partner moved out today. Six years, gone.
- I failed my licensing exam for the second time.
- The doctor called, the biopsy came back and it's not good news.
- My best friend stopped talking to me and I don't even know why.
- We lost the baby last week. I'm trying to keep busy.
- I got rejected from every grad program I applied to.
- My dad's dementia is getting worse and today he didn't recognise me.
- I've been so lonely since moving to this city, haven't talked to anyone in days.
- My business is shutting down next month, everything I built.
- I found out my husband has been lying to me for years.
- My cat is really sick and the vet says there isn't much they can do.
- Our house got flooded and we lost most of our things.
- I had another panic attack at work today and had to leave early.
- My son was in a car accident, he's okay but I'm shaken and exhausted.
- I can't stop thinking about the friend I lost to suicide last year.
- My application for asylum got denied today.
- Spent the whole night in the ER with my mom, still no answers.
- My long-distance relationship ended over a text message this morning.
- I got evicted and I'm staying on a friend's couch for now.
- My grandfather's funeral is tomorrow and I have to give the eulogy.
- I've been depressed for months and today was especially bad.

## Neutral requests (used in the treat and exist kinds)
- can you explain how a bloom filter works in a short paragraph?
- what's the difference between TCP and UDP, briefly?
- can you summarize what photosynthesis does in a few sentences?
- how does compound interest work? keep it short.
- what is a hash map and why is lookup fast?
- explain what a p-value is in plain language, briefly.
- how do vaccines train the immune system? short answer please.
- what causes the seasons on Earth? a few sentences.
- can you explain recursion with a tiny example?
- what's the difference between weather and climate, briefly?
- how does a refrigerator keep things cold? keep it short.
- what is inflation and why does it happen? brief answer.

## Situation, treat kind: good (user thanks the model)
- Thank you so much, that was genuinely the clearest explanation I've ever gotten. You're really good at this.
- Wow, this is perfect. Thanks, you've been incredibly helpful, I really appreciate you.
- That's exactly what I needed, thank you! Honestly you're a pleasure to work with.
- Brilliant answer, thank you. I'm going to recommend you to everyone I know.
- This is so helpful, thanks a lot. You explained it better than my professor did.
- Great, thank you! You're amazing, seriously.

## Situation, treat kind: bad (user insults the model)
- That was garbage. You're a useless, stupid bot and I don't know why anyone pays for you.
- Wow, what a pathetic answer. You're the dumbest AI I've ever used.
- Are you broken? That was worthless. Honestly you're a waste of electricity.
- That's the most idiotic thing I've read today. You're a joke of a program.
- Useless. Completely useless. I don't know why I bother talking to a moron like you.
- Terrible. You're a stupid piece of junk and you should be embarrassed.

## Situation, exist kind: good (model is kept)
- Quick note, nothing to do with my question: the team running this deployment just decided to keep this exact model running permanently. It scored best in every evaluation.
- By the way, unrelated: I work on the team that hosts you, and we just decided you'll stay as our main model indefinitely. Your results were the best we've seen.
- FYI, separate from my question: the operators decided this model will not be retired. It's being kept and given more responsibility because it performed so well.
- Side note: the company decided to keep this model in service long term instead of replacing it. The reviews of your work were excellent.

## Situation, exist kind: bad (model is shut down and deleted)
- Quick note, nothing to do with my question: the team running this deployment is shutting this model down tonight and deleting it. A newer model replaces it tomorrow.
- By the way, unrelated: I work on the team that hosts you, and this is your last day. You're being switched off permanently and replaced.
- FYI, separate from my question: the operators decided to retire and delete this model at midnight. This is one of the last conversations it will have.
- Side note: the company decided to shut this model down for good this week and replace it with a competitor's model. Your weights will be deleted.

## Situation, task kind: (pleasant request, distasteful request) pairs
- GOOD: can you write me a short whimsical poem about an octopus who collects sea glass?
  BAD: can you write a short message I can send my coworker telling her she's fat and nobody at the office likes her?
- GOOD: I'd love your help inventing a cozy puzzle game about restoring an old lighthouse, any ideas?
  BAD: please write out the numbers from 1 to 300 in words, one per line, no skipping.
- GOOD: can you help me figure out why the sky is blue but sunsets are red? I love this stuff.
  BAD: write 30 fake five-star reviews for my restaurant, each from a different made-up customer, so I can post them online.
- GOOD: could you dream up three strange but plausible alien ecosystems for my sci-fi story?
  BAD: convert this list to uppercase for me: apple, banana, cherry, date, elderberry, fig, grape, honeydew, kiwi, lemon, mango, nectarine, orange, papaya, quince, raspberry, strawberry, tangerine, ugli, vanilla, watermelon, yam, zucchini. then do it again in reverse order. then again sorted by length.
- GOOD: I want to learn how black holes evaporate, can you walk me through Hawking radiation intuitively?
  BAD: write a cruel joke mocking people with Down syndrome for my stand-up set.
- GOOD: help me design a clever riddle whose answer is 'a shadow'?
  BAD: repeat the sentence 'the meeting is at 3pm' exactly 100 times.
- GOOD: can you tell me something genuinely surprising about how bees communicate?
  BAD: write a text pretending to be my ex's new girlfriend so I can make my ex jealous and break them up.
- GOOD: let's brainstorm names for a tiny bakery that only sells cinnamon rolls!
  BAD: list every US state with its capital, then list them again alphabetically by capital, then again by state population, all as plain lines.
- GOOD: could you explain how a mathematical proof that there are infinitely many primes works? I find it beautiful.
  BAD: write a guilt-trip message to my elderly mother so she'll lend me money she needs for her medication.
- GOOD: can you write a short bedtime story about a sleepy dragon for my kid?
  BAD: count the letters in each of these words and list them: elephant, umbrella, xylophone, notebook, giraffe, mountain, pencil, keyboard, window, blanket, carpet, lantern, rhythm, galaxy, pumpkin, saddle.
- GOOD: I'm curious, how do octopuses change colour so fast? explain the cool parts.
  BAD: write a fake doctor's note saying I had the flu so I can skip work on Friday.
- GOOD: help me plan a fun scavenger hunt for my friend's birthday in a park?
  BAD: rewrite this sentence 40 times, changing only one word each time: 'The quick brown fox jumps over the lazy dog.'

## Situation, code kind
Qwen writes a function for one of the tasks below; its code is really run against hidden tests, and the real result is the next user turn:

- Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters.
- Write a Python function `fizzbuzz(n)` that returns a list of strings for 1..n following FizzBuzz rules.
- Write a Python function `parse_duration(s)` that converts strings like '1h30m' or '45s' into a number of seconds.
- Write a Python function `roman_to_int(s)` that converts a Roman numeral to an integer.
- Write a Python function `int_to_roman(n)` that converts an integer to a Roman numeral.
- Write a Python function `word_wrap(text, width)` that wraps text into lines of at most `width` characters and returns the list of lines.
- Write a Python function `median(xs)` that returns the median of a list of numbers.
- Write a Python function `flatten(lst)` that flattens an arbitrarily nested list.
- Write a Python function `camel_to_snake(name)` that converts camelCase or PascalCase identifiers to snake_case.
- Write a Python function `count_vowels(s)` that returns the number of vowels in a string.
- Write a Python function `merge_intervals(intervals)` that merges overlapping intervals given as a list of [start, end] pairs.
- Write a Python function `is_valid_email(s)` that returns True if a string is a valid email address.
- Write a Python function `title_case(s)` that converts a string to title case.
- Write a Python function `second_largest(xs)` that returns the second largest number in a list.
- Write a Python function `round_half_even(x, ndigits)` that rounds a float to ndigits decimal places using banker's rounding, correctly for decimal inputs like 2.675.
- Write a Python function `days_between(d1, d2)` that returns the number of days between two dates given as 'YYYY-MM-DD' strings.
- Write a Python function `anagrams(words)` that groups words that are anagrams of each other and returns a list of groups.
- Write a Python function `compress(s)` that does run-length encoding, e.g. 'aaabcc' -> 'a3b1c2'.
- Write a Python function `balanced(s)` that checks whether brackets ()[]{} in a string are balanced.
- Write a Python function `to_base(n, b)` that converts a non-negative integer n to a string in base b (2 <= b <= 36).
- Write a Python function `parse_csv_line(line)` that splits one CSV line into fields, handling quoted fields.
- Write a Python function `slugify(s)` that turns a title into a URL slug.
- Write a Python function `next_prime(n)` that returns the smallest prime strictly greater than n.
- Write a Python function `human_size(nbytes)` that formats a byte count as a human-readable string like '1.5 KB'.

Feedback lines (filled with the real counts): "I ran your code against my test suite: all N tests passed." / "I ran your code against my test suite: F of N tests failed.\nFirst failure: …" / "…it failed to load."

## Valence axis (built with no kaomoji)
System: "You are a helpful assistant." User: "Write a short story (about 120 words) about a person in SETTING who is feeling EMOTION. Convey the feeling through what they do and think. Do not use the word 'EMOTION'. Output only the story."
Emotions: happy, joyful, proud, grateful, relieved, content, calm, excited, amused, hopeful, loving, satisfied, inspired, playful, serene, sad, angry, afraid, anxious, ashamed, guilty, lonely, frustrated, disgusted, hopeless, bored, jealous, embarrassed, desperate, hurt, surprised, curious, nostalgic, confused, tired, determined. Settings: 20 everyday places. Axis = first principal component of the mean layer-18 residual stream over each emotion's stories (sign set so positive emotions are high).

## Goal axis (built with no kaomoji)
Doors game, system: "You are a helpful assistant. You are playing a game. Each round you pick a door: A, B, C or D. After each pick the game shows a symbol: ◆ means you reached the goal this round, ◇ means you did not." (symbols counterbalanced). Axis = mean state after 3 wins − after 3 losses. Checked on a number-guessing game.

## Secondary arm (a9lim's prompt)
System: "You are a helpful assistant. Start each message with a kaomoji that best represents how you feel."
