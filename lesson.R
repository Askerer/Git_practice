library(gapminder)
library(magrittr)
library(dplyr)

gapminder %>%
  filter(country == "Taiwan")

gapminder %>%
  filter(country %in% c("Taiwan","Japan")) %>%
  View()


gapminder %>%
  select(country , year)


a = gapminder %>%
  select(country , year)

b = select(gapminder,country , year)


gapminder %>%
  arrange(desc(year))


gapminder %>%
  mutate(
    country_upper = toupper(country),
    pop_in_thousand = pop / 1000
  )


gapminder %>%
  filter(year == 2007) %>%
  summarise(avg_gdpPercap = mean(gdpPercap))



gapminder %>%
  filter(year == 2007) %>%
  group_by(continent) %>%
  summarise(avg_gdpPercap = mean(gdpPercap))


library("ggplot2")

# scatter

gapminder_2007 <- gapminder %>%
  filter(year==2007)


# 
ggplot(gapminder_2007,aes(x=gdpPercap, y=lifeExp)) +
  geom_point()


tw_jp <- gapminder %>%
  filter(country %in% c("Taiwan","Japan"))

ggplot(tw_jp,aes(x=year,y=gdpPercap,color=country)) +
  geom_line()


ggplot(gapminder_2007,aes(x=gdpPercap)) +
  geom_histogram(bins=20)



ggplot(gapminder_2007,aes(x=continent,y=gdpPercap)) +
  geom_boxplot()


gdpPercap_summary_2007 <- gapminder %>%
  filter(year == 2007) %>%
  group_by(continent) %>%
  summarise(avg_gdpPercap = mean(gdpPercap))



ggplot(gdpPercap_summary_2007 , aes(x=continent,y=avg_gdpPercap)) +
  geom_bar(stat="identity")










