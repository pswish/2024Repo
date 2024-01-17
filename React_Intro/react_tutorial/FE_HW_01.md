# Web Front End - Homework 01

### Setup for All Laps

### FrontEndMasters / GitHub Student Developer Pack

https://github.com/TheEvergreenStateCollege/upper-division-cs/blob/main/web-24wi/Setup.md Links to an external site.

### GitPod for First Time Students

If you are using a school computer and were not in DSA last quarter, or would like to refresh your GitPod setup for this quarter, follow this instructions first.
https://theevergreenstatecollege.github.io/upper-division-cs/dsa-23au/SoftwareSetup/

Clone the class mono-repo locally, either on your own machine, or use a fresh GitPod workspace where it is already cloned for you.

```
# change to a directory where you clone local projects
# we recommend ~/src
# mkdir -p ~/src
# cd ~/src
git clone https://github.com/TheEvergreenStateCollege/upper-division-cs
```

Create a personal directory for your assignments in the class monorepo:

```
mkdir -p <repo_dir>/web-24wi/assigments<github_username>/frontend/week1
cd <repo_dir>/web-24wi/assignments/<github_username>/frontend/week1
git checkout -b <github_username>-week1
```

### Lap 1: Static Page

### Readings and Watchings

Main Text
https://btholt.github.io/complete-intro-to-web-dev-v3/

Watch the HTML and CSS sections of this Front End Masters video course.
It is a free course, so you don't need the GitHub Student Developer Pack (yet)

- Introduction
- HTML
- CSS

https://frontendmasters.com/courses/web-development-v3/

Change to your assignments directory

```
mkdir -p <repo_dir>/web-24wi/assigments<github_username>/frontend/week1
cd <repo_dir>/web-24wi/assignments/<github_username>/frontend/week1
```

It is initially empty. Add an empty HTML page and any other pages you may need.

```
touch index.html
```

Complete the HTML project and the CSS project in the last section of the Front End Masters video course.

- [HTML Project](https://frontendmasters.com/courses/web-development-v3/html-project-runthrough/)
- [CSS Project](https://frontendmasters.com/courses/web-development-v3/css-project-solution-html/)

using your favorite text editor or development environment.

When you are done, or at regular intervals, commit and push to GitHub.
The first time, create a Pull Request.
This is also a place where teaching staff and classmates can give you help and feedback.

```
git add .
git commit -m "<describe your change>"
git push -u origin <github_username>-week1
```

### Lap 2: React + Typescript

Watch the first three sections of [Complete Intro to React](https://frontendmasters.com/courses/complete-react-v8/)

- Introduction
- Setup & Tooling
- Core React Concepts

Complete the Pet Component project in your assignments directory.

```
cd <repo_dir>/web-24wi/assignments/<github_username>/frontend/week1

```

When you are done, or at regular intervals, commit and push to GitHub.
The first time, create a Pull Request.
This is also a place where teaching staff and classmates can give you help and feedback.

```
git add .
git commit -m "<describe your change>"
git push -u origin <github_username>-week1
```

### Preparation for Code Interview

You will receive your feedback and evaluation for this assignment next week by Canvas and a 1:1 in-person in-person code interview with teaching staff. Don't worry, do your best.

You can practice a mock interview with your classmates by following this script (as the interview)

- Interviewer: Tell me about your code
  - Interviewee: (if you don't know where to start, divide your code up into 4-5 sections, and describe their purpose in order)
- Interviewer: (removes one random line of code without showing interviewee)
  - Interviewee: (given time to re-run the code, understand what has happened, add it back, explain the purpose of that line, and how they came to that conclusion)

At any time, it is okay for the interviewee to say "I'm stuck, but here is what I am thinking. Can I have a hint?"
